"""Kết nối Ollama (API gốc của Ollama, không qua lớp tương thích OpenAI).

Dùng API gốc vì:
  - đặt được ngữ cảnh (num_ctx) cho từng lần gọi; Ollama nạp model với đúng ngữ cảnh đó
  - có số đo chính xác do chính Ollama trả về: số token prompt / sinh, thời gian đọc prompt và thời gian sinh
  - bật / tắt chế độ suy nghĩ bằng "think", ép trả JSON theo schema bằng "format"

Ollama nạp model khi có yêu cầu đầu tiên và tự gỡ sau keep_alive. Nếu một yêu cầu gửi num_ctx khác với lúc nạp,
Ollama sẽ nạp lại model, nên lớp này ghi nhớ ngữ cảnh đã chọn cho từng model và luôn gửi đúng con số đó.
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
OLLAMA_URL = os.environ.get("OLLAMA_HOST_URL") or os.environ.get("AIBENCH_OLLAMA_URL") or "http://127.0.0.1:11434"
DEFAULT_CTX = 8192  # ngữ cảnh khi chưa ai chọn (benchmark dùng mức này để các lần đo so sánh được với nhau)
KEEP_ALIVE = "30m"  # giữ model trong bộ nhớ giữa các câu hỏi / lần đo

_ctx: dict[str, int] = {}  # model -> ngữ cảnh đã chọn (dùng chung cho cả tiến trình)
_show_cache: dict[str, dict[str, Any]] = {}  # (model:digest) -> kết quả /api/show


@dataclass
class GenStats:
    """Kết quả và số đo của 1 lần sinh văn bản."""

    text: str = ""
    reasoning: str = ""  # phần "suy nghĩ" của model reasoning (gpt-oss, qwen3...)
    ttft_s: float = 0.0  # thời gian tới token đầu tiên (gồm thời gian đọc prompt)
    total_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str | None = None
    eval_s: float = 0.0  # thời gian sinh do Ollama đo (giây); 0 nếu không có

    @property
    def gen_tps(self) -> float:
        """Tốc độ sinh (token/giây): ưu tiên số đo của Ollama, không có thì tính từ thời gian sau token đầu tiên."""
        if self.eval_s > 0 and self.completion_tokens:
            return self.completion_tokens / self.eval_s
        dt = self.total_s - self.ttft_s
        return (self.completion_tokens - 1) / dt if self.completion_tokens > 1 and dt > 0 else 0.0

    @property
    def prompt_tps(self) -> float:
        """Tốc độ đọc prompt (token/giây), xấp xỉ bằng prompt_tokens / TTFT."""
        return self.prompt_tokens / self.ttft_s if self.ttft_s > 0 else 0.0


def ollama_exe() -> str | None:
    """Đường dẫn ollama.exe (PATH hoặc nơi bộ cài Windows đặt), None nếu chưa cài."""
    exe = shutil.which("ollama")
    if exe:
        return exe
    for base in (os.environ.get("LOCALAPPDATA"), os.environ.get("ProgramFiles")):
        if base:
            p = Path(base) / ("Programs/Ollama" if base == os.environ.get("LOCALAPPDATA") else "Ollama") / "ollama.exe"
            if p.is_file():
                return str(p)
    return None


async def can_think(model: str) -> bool:
    """Model có chế độ suy nghĩ (Ollama capability "thinking") không. Gửi "think" cho model không có chế độ này thì
    Ollama báo lỗi, nên chỉ gửi khi model hỗ trợ."""
    lm = Ollama()
    try:
        return "thinking" in ((await lm.show(model)).get("capabilities") or [])
    finally:
        await lm.close()


def set_ctx(model: str, ctx: int) -> None:
    _ctx[model] = int(ctx)


def ctx_of(model: str) -> int:
    return _ctx.get(model, DEFAULT_CTX)


def _to_native(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Tin nhắn kiểu OpenAI (content là list text + image_url data URL) → kiểu Ollama (content chữ + images base64)."""
    out = []
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            texts, images = [], []
            for part in c:
                if part.get("type") == "text":
                    texts.append(part.get("text", ""))
                elif part.get("type") == "image_url":
                    url = (part.get("image_url") or {}).get("url", "")
                    images.append(url.split(",", 1)[1] if url.startswith("data:") else url)
            nm = {"role": m["role"], "content": "\n".join(texts)}
            if images:
                nm["images"] = images
            out.append(nm)
        else:
            out.append({"role": m["role"], "content": c or ""})
    return out


class Ollama:
    def __init__(self, base_url: str = OLLAMA_URL):
        self.base_url = base_url.rstrip("/")
        self._http = httpx.AsyncClient(timeout=httpx.Timeout(900, connect=5))

    async def close(self) -> None:
        await self._http.aclose()

    # ---- điều khiển Ollama ----------------------------------------------------
    async def is_up(self) -> bool:
        try:
            r = await self._http.get(f"{self.base_url}/api/version", timeout=3)
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    async def version(self) -> str:
        try:
            return (await self._http.get(f"{self.base_url}/api/version", timeout=3)).json().get("version", "")
        except (httpx.HTTPError, ValueError):
            return ""

    @staticmethod
    async def start_server() -> None:
        """Mở Ollama (ứng dụng khay hệ thống, nó tự chạy server); không có thì chạy `ollama serve` ở nền."""
        exe = ollama_exe()
        if not exe:
            raise RuntimeError("Chưa cài Ollama. Tải tại https://ollama.com/download rồi chạy lại.")
        app = Path(exe).with_name("ollama app.exe")
        if app.is_file():
            os.startfile(app)  # noqa: S606
        else:
            subprocess.Popen([exe, "serve"], creationflags=_NO_WINDOW | getattr(subprocess, "DETACHED_PROCESS", 0),
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(30):
            await asyncio.sleep(1)
            if await Ollama().is_up():
                return
        raise RuntimeError("Đã mở Ollama nhưng server chưa trả lời sau 30 giây")

    async def load(self, model: str, ctx: int | None = None) -> None:
        """Nạp model vào bộ nhớ với ngữ cảnh ctx (mặc định: ngữ cảnh đã chọn trước đó cho model này)."""
        if ctx:
            set_ctx(model, ctx)
        if await self._is_embedding(model):
            r = await self._http.post(f"{self.base_url}/api/embed", json={
                "model": model, "input": ["."], "keep_alive": KEEP_ALIVE, "options": {"num_ctx": ctx_of(model)}})
        else:
            r = await self._http.post(f"{self.base_url}/api/generate", json={
                "model": model, "keep_alive": KEEP_ALIVE, "options": {"num_ctx": ctx_of(model)}})
        if r.status_code != 200:
            raise RuntimeError(f"Ollama không nạp được {model}: {r.text[:300]}")

    async def unload(self, model: str) -> None:
        if await self._is_embedding(model):
            await self._http.post(f"{self.base_url}/api/embed", json={"model": model, "input": [], "keep_alive": 0})
        else:
            await self._http.post(f"{self.base_url}/api/generate", json={"model": model, "keep_alive": 0})
        for _ in range(20):  # đợi Ollama thật sự giải phóng bộ nhớ
            if model not in {m["name"] for m in await self.ps()}:
                return
            await asyncio.sleep(0.5)

    async def unload_all(self, keep: set[str] | None = None) -> None:
        for m in await self.ps():
            if not keep or m["name"] not in keep:
                await self.unload(m["name"])

    async def ps(self) -> list[dict[str, Any]]:
        """Model đang nạp: [{name, size, size_vram, context_length?, ...}]."""
        try:
            r = await self._http.get(f"{self.base_url}/api/ps", timeout=10)
            return r.json().get("models", []) if r.status_code == 200 else []
        except (httpx.HTTPError, ValueError):
            return []

    async def show(self, model: str, digest: str = "") -> dict[str, Any]:
        key = f"{model}:{digest}"
        if key not in _show_cache:
            r = await self._http.post(f"{self.base_url}/api/show", json={"model": model}, timeout=30)
            _show_cache[key] = r.json() if r.status_code == 200 else {}
        return _show_cache[key]

    async def _is_embedding(self, model: str) -> bool:
        return "embedding" in ((await self.show(model)).get("capabilities") or [])

    async def list_models(self) -> list[dict[str, Any]]:
        """Danh sách model theo cùng định dạng app dùng: id, type (llm/vlm/embeddings), state (loaded/not-loaded),
        size_bytes, quantization, arch, max_context_length, loaded_context_length, params, capabilities."""
        r = await self._http.get(f"{self.base_url}/api/tags", timeout=15)
        r.raise_for_status()
        loaded = {m["name"]: m for m in await self.ps()}
        out = []
        for t in r.json().get("models", []):
            name, det = t.get("name") or t.get("model"), t.get("details") or {}
            info = await self.show(name, t.get("digest", ""))
            caps = info.get("capabilities") or []
            mi = info.get("model_info") or {}
            maxctx = next((v for k, v in mi.items() if k.endswith(".context_length") and isinstance(v, int)), None)
            typ = "embeddings" if "embedding" in caps else "vlm" if "vision" in caps else "llm"
            ps = loaded.get(name)
            out.append({
                "id": name, "type": typ, "state": "loaded" if ps else "not-loaded",
                "size_bytes": t.get("size"), "quantization": det.get("quantization_level"),
                "arch": det.get("family"), "params": det.get("parameter_size"),
                "max_context_length": maxctx, "capabilities": caps,
                "loaded_context_length": (ps or {}).get("context_length") or (_ctx.get(name) if ps else None),
                "size_vram": (ps or {}).get("size_vram"), "loaded_size": (ps or {}).get("size"),
                "compatibility_type": "gguf",
            })
        return out

    # ---- sinh văn bản ---------------------------------------------------------
    async def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int = 256,
        temperature: float = 0.0,
        should_stop: Callable[[GenStats], bool] | None = None,
        on_piece: Callable[[str, str], None] | None = None,
        **extra: Any,
    ) -> GenStats:
        """Hỏi model (stream) và đo TTFT, tốc độ sinh.

        `messages` theo chuẩn OpenAI (ảnh gửi dạng image_url data URL) để các module dùng chung được.
        extra hỗ trợ: response_format (json_schema → format của Ollama), reasoning_effort ("none" → tắt suy nghĩ,
        "low"/"medium"/"high" → mức suy nghĩ), think (bool). `should_stop` được gọi sau mỗi 100 mẩu; trả True thì
        ngắt giữa chừng (finish_reason = "aborted"). `on_piece(text, thinking)` nhận từng mẩu khi đang sinh.
        """
        payload: dict[str, Any] = {
            "model": model, "messages": _to_native(messages), "stream": True, "keep_alive": KEEP_ALIVE,
            "options": {"num_ctx": ctx_of(model), "num_predict": max_tokens if max_tokens and max_tokens > 0 else -1,
                        "temperature": temperature},
        }
        rf = extra.get("response_format") or {}
        if rf.get("type") == "json_schema":
            payload["format"] = (rf.get("json_schema") or {}).get("schema") or "json"
        elif rf.get("type") == "json_object":
            payload["format"] = "json"
        effort = extra.get("reasoning_effort")
        if ("think" in extra or effort) and "thinking" in ((await self.show(model)).get("capabilities") or []):
            if "think" in extra:
                payload["think"] = extra["think"]
            elif effort == "none":
                payload["think"] = False
            elif effort in ("low", "medium", "high"):
                payload["think"] = effort
        st = GenStats()
        chunks = 0
        t0 = time.perf_counter()
        first: float | None = None
        async with self._http.stream("POST", f"{self.base_url}/api/chat", json=payload) as r:
            if r.status_code != 200:
                body = (await r.aread()).decode("utf-8", "replace")
                raise RuntimeError(f"Ollama trả lỗi {r.status_code}: {body[:500]}")
            async for line in r.aiter_lines():
                if not line.strip():
                    continue
                obj = json.loads(line)
                if obj.get("error"):
                    raise RuntimeError(f"Ollama báo lỗi: {obj['error']}")
                msg = obj.get("message") or {}
                piece, think = msg.get("content") or "", msg.get("thinking") or ""
                if piece or think:
                    if first is None:
                        first = time.perf_counter()
                    chunks += 1
                    st.text += piece
                    st.reasoning += think
                    if on_piece:
                        on_piece(piece, think)
                if obj.get("done"):
                    st.finish_reason = obj.get("done_reason") or "stop"
                    st.prompt_tokens = obj.get("prompt_eval_count") or 0
                    st.completion_tokens = obj.get("eval_count") or 0
                    st.eval_s = (obj.get("eval_duration") or 0) / 1e9
                    break
                if should_stop and chunks and chunks % 100 == 0 and should_stop(st):
                    st.finish_reason = "aborted"
                    break  # ra khỏi `async with` là đóng kết nối, Ollama dừng sinh
        end = time.perf_counter()
        st.total_s = end - t0
        st.ttft_s = (first or end) - t0
        if not st.completion_tokens:
            st.completion_tokens = chunks
        return st

    async def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        r = await self._http.post(f"{self.base_url}/api/embed", json={
            "model": model, "input": texts, "truncate": True, "keep_alive": KEEP_ALIVE,
            "options": {"num_ctx": ctx_of(model)}})
        if r.status_code != 200:
            raise RuntimeError(f"Ollama trả lỗi {r.status_code} khi embedding: {r.text[:300]}")
        return r.json().get("embeddings") or []

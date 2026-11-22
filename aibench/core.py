"""Lõi dùng chung: định nghĩa Module, Context, nạp module tự động, chạy và lưu kết quả.

Thêm tính năng mới = thêm 1 file trong aibench/modules/ có biến MODULE (xem README).
"""
from __future__ import annotations

import asyncio
import importlib
import json
import os
import pkgutil
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from .engine import detect
from .gpu import GpuMonitor, gpu_info
from .ollama import OLLAMA_URL, Ollama

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
LM_URL = OLLAMA_URL  # địa chỉ server Ollama (đổi bằng biến môi trường AIBENCH_OLLAMA_URL)
GPU_INDEX = int(os.environ.get("AIBENCH_GPU_INDEX", "0"))


@dataclass
class Param:
    """Một tham số của module. Giao diện web tự vẽ form từ danh sách này.

    type: "model" (chọn model Ollama) | "int" | "bool" | "text" | "textarea" | "select"
    """

    key: str
    label: str
    type: str = "text"
    default: Any = None
    help: str = ""
    options: list[str] | None = None
    min: int | None = None
    max: int | None = None
    # Với type "int": mỗi giá trị thành 1 nút "Chạy <n> <unit>" trên giao diện (điền số rồi chạy luôn).
    presets: list[int] | None = None
    unit: str = ""


@dataclass
class Context:
    """Những thứ module được dùng khi chạy."""

    lm: Ollama
    gpu: GpuMonitor
    log: Callable[[str], None]
    progress: Callable[[float], None]  # 0.0 → 1.0
    # Đưa 1 ảnh lên giao diện ngay khi đang chạy (cùng định dạng với phần tử của "images").
    # Gọi lại với cùng "key" để cập nhật ảnh đó (vd. từ "đang xem" sang "đã trả lời").
    show: Callable[[dict[str, Any]], None] = lambda item: None


class Module:
    """Lớp cha của mọi module (benchmark, phân tích email, chat, X-quang...).

    run() trả về dict theo định dạng chung để giao diện hiển thị được ngay:
      {
        "summary": [{"label": str, "value": Any, "unit": str}],
        "tables":  [{"title": str, "columns": [str], "rows": [[Any]]}],
        "charts":  [{"title": str, "x": [float], "series": [{"name": str, "unit": str, "data": [float]}]}],
        "text":    str,   # tuỳ chọn: văn bản trả lời (cho module chat/phân tích)
        "images":  [{"key": str, "src": data URL, "title": str, "status": "pending"|"done", "meta": str,
                     "checks": [{"label": str, "expected": str, "got": str|None, "ok": bool|None}]}],
        "files":   [{"label": str, "path": str}],  # tuỳ chọn: file đã xuất (giao diện có nút "Mở file")
        "raw":     dict,  # tuỳ chọn: dữ liệu chi tiết
      }
    """

    id: str = ""
    name: str = ""
    description: str = ""
    category: str = "benchmark"
    params: list[Param] = []

    async def run(self, ctx: Context, params: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def describe(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "params": [asdict(p) for p in self.params],
        }

    def coerce(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Ép kiểu và kiểm tra tham số người dùng gửi lên."""
        out: dict[str, Any] = {}
        for p in self.params:
            v = raw.get(p.key, p.default)
            if p.type == "int":
                v = int(v)
                if p.min is not None:
                    v = max(p.min, v)
                if p.max is not None:
                    v = min(p.max, v)
            elif p.type == "bool":
                v = v if isinstance(v, bool) else str(v).strip().lower() in ("1", "true", "yes", "on")
            elif p.type == "select" and p.options and v not in p.options:
                raise ValueError(f"Giá trị '{v}' không hợp lệ cho '{p.label}'")
            elif p.type == "model" and not v:
                raise ValueError("Chưa chọn model")
            out[p.key] = v
        return out


async def ensure_loaded(ctx: Context, model: str, unload_others: bool = True) -> tuple[dict[str, Any], float | None]:
    """Đảm bảo model đã nạp. Trả về (thông tin model, thời gian nạp tính bằng giây hoặc None nếu đã nạp sẵn)."""
    info = next((m for m in await ctx.lm.list_models() if m.get("id") == model), None)
    if info is None:
        raise RuntimeError(f"Ollama không có model '{model}'. Tải bằng: ollama pull {model}")
    if info.get("state") == "loaded":
        ctx.log(f"Model {model} đã nạp sẵn (bỏ qua đo thời gian nạp)")
        return info, None
    if unload_others:
        ctx.log("Gỡ các model đang nạp...")
        await ctx.lm.unload_all()
    ctx.log(f"Nạp model {model}...")
    t = time.perf_counter()
    await ctx.lm.load(model)
    load_s = round(time.perf_counter() - t, 2)
    ctx.log(f"Nạp xong sau {load_s:.1f} giây")
    return info, load_s


_registry: dict[str, Module] | None = None


def modules() -> dict[str, Module]:
    """Tự động nạp mọi file trong aibench/modules/ có biến MODULE."""
    global _registry
    if _registry is None:
        from . import modules as pkg

        _registry = {}
        for m in pkgutil.iter_modules(pkg.__path__):
            mod = importlib.import_module(f"{pkg.__name__}.{m.name}")
            inst = getattr(mod, "MODULE", None)
            if isinstance(inst, Module) and inst.id:
                _registry[inst.id] = inst
    return _registry


async def run_module(
    module_id: str,
    raw_params: dict[str, Any],
    log: Callable[[str], None] = print,
    progress: Callable[[float], None] = lambda f: None,
    show: Callable[[dict[str, Any]], None] = lambda item: None,
) -> dict[str, Any]:
    """Chạy 1 module, lưu kết quả vào results/ và trả về bản ghi."""
    mod = modules().get(module_id)
    if mod is None:
        raise ValueError(f"Không có module '{module_id}'")
    params = mod.coerce(raw_params)

    lm = Ollama(LM_URL)
    gpu = GpuMonitor(GPU_INDEX)
    started = datetime.now()
    t0 = time.perf_counter()
    gpu.start()
    try:
        result = await mod.run(Context(lm=lm, gpu=gpu, log=log, progress=progress, show=show), params)
        # Mọi module đều được ghi lượng điện GPU tiêu thụ của cả lần chạy, không cần tự làm.
        total = gpu.summary()
        if total["energy_wh"] is not None:
            result.setdefault("summary", []).append(
                {"label": "Điện GPU tiêu thụ (cả lần chạy)", "value": round(total["energy_wh"], 2), "unit": "Wh"}
            )
            result.setdefault("raw", {})["gpu_whole_run"] = total
        # Ghi lại model đã chạy bằng GPU hay CPU (hỏi khi model còn đang nạp).
        engine = await detect()
        # Ghi lại tham số nén (vd. Q4_K_M, MXFP4) để về sau vẫn biết lần chạy này dùng bản nào.
        model_info: dict[str, Any] = {}
        try:
            info = next((m for m in await lm.list_models() if m.get("id") == params.get("model")), None) or {}
            model_info = {k: info[k] for k in ("quantization", "arch", "type", "compatibility_type") if info.get(k)}
        except Exception:  # noqa: BLE001 - thiếu thông tin này không được làm hỏng kết quả đo
            pass
    finally:
        gpu.stop()
        await lm.close()

    record = {
        "id": started.strftime("%Y%m%d-%H%M%S") + "_" + module_id,
        "module": module_id,
        "module_name": mod.name,
        "params": params,
        "started_at": started.isoformat(timespec="seconds"),
        "duration_s": round(time.perf_counter() - t0, 1),
        "gpu": gpu_info(GPU_INDEX),
        "engine": engine,
        "model_info": model_info,
        "result": result,
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / f"{record['id']}.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    progress(1.0)
    return record


def list_results() -> list[dict[str, Any]]:
    """Danh sách kết quả đã lưu (mới nhất trước), chỉ gồm phần tóm tắt."""
    out = []
    for f in sorted(RESULTS_DIR.glob("*.json"), reverse=True):
        try:
            r = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        out.append(
            {
                "id": r.get("id", f.stem),
                "module": r.get("module"),
                "module_name": r.get("module_name"),
                "model": (r.get("params") or {}).get("model"),
                "quant": (r.get("model_info") or {}).get("quantization"),
                "started_at": r.get("started_at"),
                "duration_s": r.get("duration_s"),
                "gpu": (r.get("gpu") or {}).get("name"),
                "summary": (r.get("result") or {}).get("summary", [])[:5],
            }
        )
    return out


def load_result(result_id: str) -> dict[str, Any] | None:
    if not re.fullmatch(r"[\w\-]+", result_id):
        return None
    f = RESULTS_DIR / f"{result_id}.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.is_file() else None

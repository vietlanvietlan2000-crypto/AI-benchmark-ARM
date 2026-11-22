"""Model đang chạy bằng GPU, tràn một phần sang CPU, hay chỉ CPU.

Ollama báo thẳng qua /api/ps: `size` là tổng bộ nhớ model đang chiếm, `size_vram` là phần nằm trên GPU.
  - size_vram ≈ size  → chạy hoàn toàn bằng GPU
  - 0 < size_vram < size → một phần lớp model chạy bằng CPU ("tràn"), chậm hơn nhiều
  - size_vram = 0     → chỉ CPU (vd. bản Ollama không dùng được GPU trên máy này)

Trên máy bộ nhớ chung (RTX Spark, GPU tích hợp dùng chung RAM với CPU) không có "VRAM" riêng; Ollama vẫn báo
phần model giao cho GPU xử lý trong size_vram.
"""
from __future__ import annotations

import asyncio
import sys
import time
from typing import Any

_MB = 1024 * 1024
TTL_S = 5


def ram_total_mb() -> int | None:
    """Tổng RAM của máy (MB). None nếu không hỏi được."""
    st = _mem()
    return round(st[0] / _MB) if st else None


def ram_used_mb() -> int | None:
    st = _mem()
    return round((st[0] - st[1]) / _MB) if st else None


def _mem() -> tuple[int, int] | None:
    """(tổng, còn trống) RAM tính bằng byte."""
    if sys.platform != "win32":
        try:
            info = {ln.split(":")[0]: int(ln.split()[1]) * 1024 for ln in open("/proc/meminfo", encoding="ascii")}
            return info["MemTotal"], info.get("MemAvailable", info.get("MemFree", 0))
        except (OSError, KeyError, ValueError, IndexError):
            return None
    import ctypes

    class _Mem(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
            (n, ctypes.c_ulonglong) for n in ("total", "avail", "page_total", "page_avail", "virt_total", "virt_avail", "ext")]

    st = _Mem()
    st.length = ctypes.sizeof(st)
    try:
        ok = ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
    except (OSError, AttributeError):
        return None
    return (st.total, st.avail) if ok else None


async def detect() -> dict[str, Any]:
    """Trả {"mode": "gpu"|"spill"|"cpu"|"idle"|"unknown", ...} cho model đang nạp (model lớn nhất nếu có nhiều)."""
    from .ollama import Ollama, ctx_of

    lm = Ollama()
    try:
        if not await lm.is_up():
            return {"mode": "unknown"}
        ps = await lm.ps()
    finally:
        await lm.close()
    if not ps:
        return {"mode": "idle"}
    m = max(ps, key=lambda x: x.get("size") or 0)
    size, vram = m.get("size") or 0, m.get("size_vram") or 0
    if vram <= 0:
        mode = "cpu"
    elif vram < size * 0.97:
        mode = "spill"
    else:
        mode = "gpu"
    return {"mode": mode, "backend": "GPU" if vram else "CPU", "model": m.get("name"), "gpu_layers": None,
            "ctx": m.get("context_length") or ctx_of(m.get("name", "")), "vram_mb": round(vram / _MB),
            "shared_mb": round((size - vram) / _MB) if mode == "spill" else 0, "ram_mb": round(size / _MB),
            "gpu_pct": round(vram / size * 100) if size else 0}


_cache: dict[str, Any] = {"t": 0.0, "v": {"mode": "unknown"}}
_lock = asyncio.Lock()


async def engine_status(fresh: bool = False) -> dict[str, Any]:
    """Như detect() nhưng có nhớ tạm, dùng cho /api/status (được hỏi vài giây 1 lần)."""
    async with _lock:
        if fresh or time.monotonic() - _cache["t"] > TTL_S:
            _cache["v"] = await detect()
            _cache["t"] = time.monotonic()
        return _cache["v"]

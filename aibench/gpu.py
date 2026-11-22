"""Giám sát GPU NVIDIA bằng nvidia-smi (không cần cài thêm thư viện)."""
from __future__ import annotations

import shutil
import subprocess
import threading
import time
from typing import Any

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
_FIELDS = "utilization.gpu,memory.used,temperature.gpu,power.draw"


def _num(s: str) -> float | None:
    try:
        return float(s.strip())
    except ValueError:  # "[N/A]"
        return None


def gpu_info(index: int = 0) -> dict[str, Any]:
    """Tên GPU, tổng VRAM (MB), phiên bản driver. Trả {} nếu không có nvidia-smi."""
    if not shutil.which("nvidia-smi"):
        return {}
    try:
        out = subprocess.run(
            ["nvidia-smi", "-i", str(index), "--query-gpu=name,memory.total,memory.used,driver_version,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, creationflags=_NO_WINDOW,
        ).stdout.strip()
        name, total, used, driver, power = [x.strip() for x in out.splitlines()[0].split(",")]
        info = {"name": name, "vram_total_mb": _num(total), "vram_used_mb": _num(used), "driver": driver,
                "power_w": _num(power), "unified": False}
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return {}
    if info["vram_total_mb"] is None:
        # GPU tích hợp dùng chung bộ nhớ với CPU (RTX Spark…): nvidia-smi không báo VRAM riêng ("[N/A]").
        # Bộ nhớ GPU dùng được chính là RAM hệ thống, nên báo tổng / đã dùng của RAM.
        from .engine import ram_total_mb, ram_used_mb

        info.update(vram_total_mb=ram_total_mb(), vram_used_mb=ram_used_mb(), unified=True)
    return info


class GpuMonitor:
    """Lấy mẫu GPU liên tục ở luồng nền.

    Mỗi mẫu: {"t": giây kể từ lúc start, "util": %, "vram": MB, "temp": °C, "power": W}
    """

    def __init__(self, index: int = 0, interval_ms: int = 500):
        self.index = index
        self.interval_ms = interval_ms
        self.samples: list[dict[str, float | None]] = []
        self._proc: subprocess.Popen | None = None
        self._t0 = 0.0

    @property
    def available(self) -> bool:
        return self._proc is not None

    def start(self) -> None:
        if not shutil.which("nvidia-smi"):
            return
        self._t0 = time.perf_counter()
        self._proc = subprocess.Popen(
            ["nvidia-smi", "-i", str(self.index), f"--query-gpu={_FIELDS}",
             "--format=csv,noheader,nounits", "-lms", str(self.interval_ms)],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, creationflags=_NO_WINDOW,
        )
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        assert self._proc and self._proc.stdout
        from .engine import ram_used_mb

        for line in self._proc.stdout:
            parts = line.split(",")
            if len(parts) != 4:
                continue
            util, vram, temp, power = (_num(p) for p in parts)
            if vram is None:  # bộ nhớ chung: lấy RAM đang dùng của cả máy
                vram = ram_used_mb()
            self.samples.append(
                {"t": round(time.perf_counter() - self._t0, 2), "util": util, "vram": vram, "temp": temp, "power": power}
            )

    def stop(self) -> None:
        if self._proc:
            self._proc.terminate()
            self._proc = None

    def mark(self) -> int:
        """Đánh dấu vị trí hiện tại, dùng với summary() để thống kê riêng 1 giai đoạn."""
        return len(self.samples)

    def summary(self, since: int = 0) -> dict[str, float | None]:
        """Thống kê các mẫu từ vị trí `since`."""
        s = self.samples[since:]

        def col(k: str) -> list[float]:
            return [x[k] for x in s if x[k] is not None]

        def avg(v: list[float]) -> float | None:
            return round(sum(v) / len(v), 1) if v else None

        def mx(v: list[float]) -> float | None:
            return max(v) if v else None

        return {
            "util_avg": avg(col("util")),
            "util_max": mx(col("util")),
            "vram_peak_mb": mx(col("vram")),
            "temp_max": mx(col("temp")),
            "power_avg_w": avg(col("power")),
            "power_max_w": mx(col("power")),
            "energy_wh": self.energy_wh(since),
        }

    def energy_wh(self, since: int = 0) -> float | None:
        """Điện năng GPU đã tiêu thụ (Wh) từ vị trí `since`: cộng dồn công suất × thời gian giữa các mẫu."""
        s = [x for x in self.samples[max(0, since - 1):] if x["power"] is not None]
        if len(s) < 2:
            return None
        joules = sum((b["t"] - a["t"]) * (a["power"] + b["power"]) / 2 for a, b in zip(s, s[1:]))
        return round(joules / 3600, 3)

    def chart(self, max_points: int = 300) -> dict[str, Any]:
        """Dữ liệu biểu đồ theo định dạng chung của Module.run()."""
        s = self.samples
        step = max(1, len(s) // max_points)
        s = s[::step]
        return {
            "title": "GPU theo thời gian",
            "x": [x["t"] for x in s],
            "series": [
                {"name": "GPU sử dụng", "unit": "%", "data": [x["util"] for x in s]},
                {"name": "VRAM", "unit": "MB", "data": [x["vram"] for x in s]},
                {"name": "Nhiệt độ", "unit": "°C", "data": [x["temp"] for x in s]},
                {"name": "Công suất", "unit": "W", "data": [x["power"] for x in s]},
            ],
        }

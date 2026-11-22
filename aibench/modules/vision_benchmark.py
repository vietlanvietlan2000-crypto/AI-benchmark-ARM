"""Benchmark phân tích hình ảnh: đo tốc độ VÀ độ chính xác của GPU + model thị giác.

Phần mềm tự vẽ ảnh test ở nhiều độ phân giải. Mỗi ảnh chứa nội dung đã biết trước:
  - vài hình tròn màu (model phải đếm)
  - một mã lớn 6 ký tự (model phải đọc)
  - một dãy số nhỏ ở đáy ảnh (kiểm tra khả năng nhìn chi tiết nhỏ)
Vì biết trước đáp án nên chấm được đúng/sai, và mỗi ảnh là ngẫu nhiên nên không bị cache.

Điểm tổng = số ảnh xử lý mỗi phút × tỉ lệ trả lời đúng.
"""
from __future__ import annotations

import base64
import io
import json
import random
import re
import statistics
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from ..core import Context, Module, Param, ensure_loaded

_COLORS = ["#dc2626", "#2563eb", "#16a34a", "#d97706", "#7c3aed", "#0891b2"]
_CODE_CHARS = "ABCDEFGHKLMNPRSTUVXYZ23456789"  # bỏ các ký tự dễ nhầm (O/0, I/1, J, Q, W)
_PROMPT = (
    "Trong ảnh có: một số hình tròn màu, một mã lớn gồm 6 ký tự, và một dãy 5 chữ số nhỏ ở dưới cùng. "
    'Trả lời đúng một dòng JSON, không giải thích: {"circles": <số hình tròn>, "code": "<mã lớn>", "small": "<dãy số nhỏ>"}'
)


def _font(px: int) -> ImageFont.ImageFont:
    for name in ("arialbd.ttf", "arial.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, px)
        except OSError:
            continue
    return ImageFont.load_default(px)


def make_image(size: int, rng: random.Random) -> tuple[str, dict[str, Any]]:
    """Vẽ ảnh test vuông cạnh `size`. Trả về (data URL dạng PNG, đáp án)."""
    truth = {
        "circles": rng.randint(3, 8),
        "code": "".join(rng.choices(_CODE_CHARS, k=6)),
        "small": "".join(rng.choices("0123456789", k=5)),
    }
    img = Image.new("RGB", (size, size), "white")
    d = ImageDraw.Draw(img)
    # hình tròn: xếp ngẫu nhiên vào lưới 4×2 ở nửa trên
    w, h, r = size / 4, size * 0.25, size * 0.09
    for cell in rng.sample(range(8), truth["circles"]):
        cx, cy = (cell % 4 + 0.5) * w, size * 0.04 + (cell // 4 + 0.5) * h
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=rng.choice(_COLORS))
    d.text((size / 2, size * 0.70), truth["code"], font=_font(int(size * 0.14)), fill="black", anchor="mm")
    d.text((size / 2, size * 0.93), truth["small"], font=_font(max(8, int(size * 0.025))), fill="black", anchor="mm")
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(), truth


_LABELS = {"circles": "Đếm hình tròn", "code": "Mã lớn", "small": "Số nhỏ"}


def parse_answer(answer: str) -> dict[str, Any]:
    """Lấy JSON cuối cùng trong câu trả lời của model ({} nếu không có)."""
    for m in reversed(re.findall(r"\{[^{}]*\}", answer)):
        try:
            got = json.loads(m)
            if isinstance(got, dict):
                return got
        except ValueError:
            continue
    return {}


def make_checks(truth: dict[str, Any], got: dict[str, Any] | None) -> list[dict[str, Any]]:
    """So câu trả lời với đáp án. got=None nghĩa là model chưa trả lời (ảnh đang được xem)."""

    def norm(v: Any) -> str:
        return re.sub(r"[\s\-]", "", str(v)).upper()

    return [
        {
            "key": k,
            "label": _LABELS[k],
            "expected": str(truth[k]),
            "got": None if got is None else str(got.get(k, "(không trả lời)")),
            "ok": None if got is None else norm(got.get(k, "")) == norm(truth[k]),
        }
        for k in truth
    ]


def grade(answer: str, truth: dict[str, Any]) -> dict[str, bool]:
    """Chấm câu trả lời của model theo đáp án."""
    return {c["key"]: c["ok"] for c in make_checks(truth, parse_answer(answer))}


class VisionBenchmark(Module):
    id = "vision_benchmark"
    name = "Benchmark phân tích hình ảnh"
    description = "Gửi ảnh test ở nhiều độ phân giải tới model thị giác; đo thời gian xử lý ảnh, tốc độ sinh và độ chính xác (đếm hình, đọc chữ)."
    category = "benchmark"
    params = [
        Param("model", "Model thị giác", "model", help="Chỉ hiện các model đọc được ảnh", options=["vlm"]),
        Param("resolutions", "Độ phân giải (px, cách nhau dấu phẩy)", "text", "448,896,1344,2048",
              "Ảnh vuông; ảnh lớn có thể bị Ollama thu nhỏ trước khi đưa vào model"),
        Param("runs", "Số ảnh mỗi độ phân giải", "int", 3, "Mỗi ảnh có nội dung ngẫu nhiên khác nhau", min=1, max=20),
        Param("unload_others", "Gỡ các model khác trước khi nạp", "bool", True, "Giải phóng VRAM để đo chính xác"),
    ]

    def coerce(self, raw: dict[str, Any]) -> dict[str, Any]:
        p = super().coerce(raw)
        sizes = [int(x) for x in re.findall(r"\d+", str(p["resolutions"]))]
        sizes = sorted({min(4096, max(128, s)) for s in sizes})
        if not sizes:
            raise ValueError("Cần ít nhất một độ phân giải, ví dụ: 448,896")
        p["resolutions"] = ",".join(map(str, sizes))
        return p

    async def run(self, ctx: Context, p: dict[str, Any]) -> dict[str, Any]:
        model, runs = p["model"], p["runs"]
        sizes = [int(x) for x in p["resolutions"].split(",")]
        total_steps = 2 + len(sizes) * runs
        done = 0

        def step() -> None:
            nonlocal done
            done += 1
            ctx.progress(done / total_steps)

        async def ask(url: str):
            return await ctx.lm.chat(
                model,
                [{"role": "user", "content": [{"type": "text", "text": _PROMPT},
                                              {"type": "image_url", "image_url": {"url": url}}]}],
                max_tokens=1024,
            )

        rng = random.Random()
        info, load_s = await ensure_loaded(ctx, model, p["unload_others"])
        start = ctx.gpu.mark()  # bỏ qua số đo GPU của model cũ trước khi gỡ
        step()
        ctx.log("Chạy làm nóng (nạp bộ mã hoá ảnh)...")
        await ask(make_image(sizes[0], rng)[0])
        step()

        rows: list[dict[str, Any]] = []
        gallery: list[dict[str, Any]] = []
        for size in sizes:
            mark = ctx.gpu.mark()
            items = []
            for i in range(runs):
                url, truth = make_image(size, rng)
                card = {"key": f"{size}-{i}", "src": url, "title": f"{size}×{size} · ảnh {i + 1}/{runs}",
                        "status": "pending", "checks": make_checks(truth, None)}
                ctx.show(card)  # hiện ảnh lên giao diện trước khi model trả lời
                s = await ask(url)
                checks = make_checks(truth, parse_answer(s.text))
                ok = {c["key"]: c["ok"] for c in checks}
                card = {**card, "status": "done", "checks": checks,
                        "meta": f"{s.total_s:.1f} giây · ảnh chiếm {s.prompt_tokens} token · {s.gen_tps:.0f} tok/s"}
                ctx.show(card)
                gallery.append(card)
                items.append({"stats": s, "ok": ok, "truth": truth, "answer": s.text.strip()[:300]})
                marks = "".join("✓" if v else "✗" for v in ok.values())
                ctx.log(f"{size}px [{i + 1}/{runs}]: {s.total_s:.1f} giây, TTFT {s.ttft_s * 1000:.0f} ms, "
                        f"{s.prompt_tokens} token vào, đếm/mã/số nhỏ: {marks}")
                step()
            st = [x["stats"] for x in items]
            rows.append({
                "size": size,
                "prompt_tokens": round(statistics.median(s.prompt_tokens for s in st)),
                "ttft_ms": statistics.median(s.ttft_s * 1000 for s in st),
                "total_s": statistics.median(s.total_s for s in st),
                "gen_tps": statistics.median(s.gen_tps for s in st),
                "correct": {k: sum(x["ok"][k] for x in items) for k in ("circles", "code", "small")},
                "gpu": ctx.gpu.summary(mark),
                "runs": [{"truth": x["truth"], "answer": x["answer"], "ok": x["ok"],
                          "ttft_s": round(x["stats"].ttft_s, 3), "total_s": round(x["stats"].total_s, 3),
                          "completion_tokens": x["stats"].completion_tokens} for x in items],
            })

        # --- tổng hợp --------------------------------------------------------
        n = len(sizes) * runs
        checks = n * 3
        right = sum(sum(r["correct"].values()) for r in rows)
        accuracy = right / checks
        sec_per_image = statistics.mean(x["total_s"] for r in rows for x in r["runs"])
        per_min = 60 / sec_per_image
        overall = ctx.gpu.summary(start)

        def pct(key: str) -> float:
            return round(100 * sum(r["correct"][key] for r in rows) / n)

        summary = [
            {"label": "Điểm tổng", "value": round(per_min * accuracy, 1), "unit": "điểm"},
            {"label": "Tốc độ xử lý", "value": round(per_min, 1), "unit": "ảnh/phút"},
            {"label": "Độ chính xác", "value": round(100 * accuracy), "unit": "%"},
            {"label": "Đếm hình đúng", "value": pct("circles"), "unit": "%"},
            {"label": "Đọc mã lớn đúng", "value": pct("code"), "unit": "%"},
            {"label": "Đọc số nhỏ đúng", "value": pct("small"), "unit": "%"},
        ]
        if ctx.gpu.available:
            summary += [
                {"label": "VRAM đỉnh", "value": overall["vram_peak_mb"], "unit": "MB"},
                {"label": "GPU sử dụng TB", "value": overall["util_avg"], "unit": "%"},
                {"label": "Nhiệt độ cao nhất", "value": overall["temp_max"], "unit": "°C"},
                {"label": "Công suất TB", "value": overall["power_avg_w"], "unit": "W"},
            ]
            measured = sum(r["gpu"]["energy_wh"] or 0 for r in rows)
            if measured:
                summary.append({"label": "Điện mỗi ảnh", "value": round(measured / n, 3), "unit": "Wh"})
        if load_s is not None:
            summary.append({"label": "Thời gian nạp model", "value": round(load_s, 1), "unit": "giây"})

        table = {
            "title": "Chi tiết theo độ phân giải (trung vị)",
            "columns": ["Ảnh", "Token vào", "Tới token đầu (ms)", "Tổng (giây)", "Sinh (tok/s)",
                        "Đếm hình", "Mã lớn", "Số nhỏ", "GPU TB (%)", "VRAM đỉnh (MB)", "Công suất TB (W)",
                        f"Điện {runs} ảnh (Wh)"],
            "rows": [
                [f"{r['size']}×{r['size']}", r["prompt_tokens"], round(r["ttft_ms"]), round(r["total_s"], 2),
                 round(r["gen_tps"], 1), f"{r['correct']['circles']}/{runs}", f"{r['correct']['code']}/{runs}",
                 f"{r['correct']['small']}/{runs}", r["gpu"]["util_avg"], r["gpu"]["vram_peak_mb"],
                 r["gpu"]["power_avg_w"], r["gpu"]["energy_wh"]]
                for r in rows
            ],
        }
        return {
            "summary": summary,
            "images": gallery,
            "tables": [table],
            "charts": [ctx.gpu.chart()] if ctx.gpu.available else [],
            "raw": {"model_info": info, "load_time_s": load_s, "resolutions": rows, "gpu_overall": overall},
        }


MODULE = VisionBenchmark()

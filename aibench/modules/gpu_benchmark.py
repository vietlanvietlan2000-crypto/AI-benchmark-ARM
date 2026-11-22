"""Benchmark hiệu năng GPU khi chạy LLM qua Ollama.

4 kịch bản:
  1. Sinh ngắn      - prompt ngắn, đo tốc độ sinh và độ trễ token đầu (TTFT)
  2. Ngữ cảnh dài   - prompt ~3000 token, đo tốc độ GPU đọc prompt
  3. Sinh dài       - sinh liên tục, đo tốc độ khi GPU chạy ổn định + nhiệt/công suất
  4. Song song      - nhiều yêu cầu cùng lúc, đo tổng thông lượng

Điểm tổng = 10 × trung bình nhân (tok/s sinh ngắn, tok/s sinh dài, tok/s song song, tok/s đọc prompt ÷ 10).
"""
from __future__ import annotations

import asyncio
import math
import statistics
import time
import uuid
from typing import Any

from ..core import Context, Module, Param, ensure_loaded
from ..ollama import GenStats

_SHORT = "Hãy viết một bài luận thật dài và chi tiết về lịch sử phát triển của máy tính, từ bàn tính đến trí tuệ nhân tạo."
_LONG_GEN = "Hãy viết một câu chuyện khoa học viễn tưởng thật dài, nhiều chương, kể về một con tàu thám hiểm sao Hỏa. Viết liên tục, không dừng lại."
_FILLER = [
    "Bộ xử lý đồ họa ban đầu được thiết kế để dựng hình ảnh ba chiều trong trò chơi điện tử.",
    "Nhờ khả năng tính toán song song trên hàng nghìn nhân, GPU trở thành nền tảng của học sâu.",
    "Một mô hình ngôn ngữ lớn dự đoán token tiếp theo dựa trên toàn bộ ngữ cảnh phía trước.",
    "Dung lượng VRAM quyết định kích thước mô hình và độ dài ngữ cảnh có thể nạp hoàn toàn lên GPU.",
    "Lượng tử hóa giảm số bit của trọng số, giúp mô hình nhỏ hơn và chạy nhanh hơn với sai số chấp nhận được.",
    "Băng thông bộ nhớ thường là nút thắt cổ chai khi sinh token, hơn là sức mạnh tính toán thuần túy.",
    "Nhiệt độ và công suất tiêu thụ ảnh hưởng tới khả năng duy trì hiệu năng trong thời gian dài.",
]


def _long_prompt(run: int, target_chars: int = 9000) -> str:
    # Mã ngẫu nhiên ở đầu để Ollama không dùng lại cache prompt của lần chạy trước.
    parts = [f"Mã phiên: {uuid.uuid4().hex}-{run}. Dưới đây là một tài liệu kỹ thuật.\n"]
    i = 0
    while sum(len(p) for p in parts) < target_chars:
        parts.append(f"{i + 1}. {_FILLER[i % len(_FILLER)]}")
        i += 1
    parts.append("\nHãy tóm tắt tài liệu trên trong đúng 2 câu.")
    return "\n".join(parts)


def _user(text: str) -> list[dict[str, Any]]:
    return [{"role": "user", "content": text}]


def _med(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def _r(v: float | None, nd: int = 1) -> float | None:
    return None if v is None else round(v, nd)


class GpuBenchmark(Module):
    id = "gpu_benchmark"
    name = "Benchmark GPU (LLM)"
    description = "Đo tốc độ sinh token, độ trễ, tốc độ đọc prompt, thông lượng song song, VRAM, nhiệt độ và công suất của GPU."
    category = "benchmark"
    params = [
        Param("model", "Model", "model", help="Model LLM trong Ollama dùng để đo"),
        Param("runs", "Số lần lặp mỗi kịch bản", "int", 3, "Lấy trung vị để giảm nhiễu", min=1, max=10),
        Param("max_tokens", "Số token sinh (kịch bản ngắn)", "int", 256, "Kịch bản sinh dài dùng gấp 3 lần", min=32, max=2048),
        Param("concurrency", "Số yêu cầu song song", "int", 4, "Đặt 1 để bỏ qua kịch bản song song", min=1, max=16),
        Param("unload_others", "Gỡ các model khác trước khi nạp", "bool", True, "Giải phóng VRAM để đo chính xác"),
    ]

    async def run(self, ctx: Context, p: dict[str, Any]) -> dict[str, Any]:
        model, runs, max_tokens, conc = p["model"], p["runs"], p["max_tokens"], p["concurrency"]
        total_steps = 2 + runs * 3 + (1 if conc > 1 else 0)
        done = 0

        def step() -> None:
            nonlocal done
            done += 1
            ctx.progress(done / total_steps)

        # --- nạp model -------------------------------------------------------
        info, load_s = await ensure_loaded(ctx, model, p["unload_others"])
        start = ctx.gpu.mark()  # bỏ qua số đo GPU của model cũ trước khi gỡ
        step()

        ctx.log("Chạy làm nóng...")
        await ctx.lm.chat(model, _user("Xin chào"), max_tokens=16)
        step()

        async def scenario(title: str, make_prompt, max_tok: int) -> dict[str, Any]:
            mark = ctx.gpu.mark()
            stats: list[GenStats] = []
            for i in range(runs):
                s = await ctx.lm.chat(model, _user(make_prompt(i)), max_tokens=max_tok)
                stats.append(s)
                ctx.log(
                    f"{title} [{i + 1}/{runs}]: {s.gen_tps:.1f} tok/s, TTFT {s.ttft_s * 1000:.0f} ms, "
                    f"{s.prompt_tokens} → {s.completion_tokens} token"
                )
                step()
            return {
                "title": title,
                "prompt_tokens": round(_med([s.prompt_tokens for s in stats])),
                "completion_tokens": round(_med([s.completion_tokens for s in stats])),
                "tokens_total": sum(s.completion_tokens for s in stats),
                "ttft_ms": _med([s.ttft_s * 1000 for s in stats]),
                "gen_tps": _med([s.gen_tps for s in stats]),
                "prompt_tps": _med([s.prompt_tps for s in stats]),
                "gpu": ctx.gpu.summary(mark),
                "runs": [
                    {"ttft_s": round(s.ttft_s, 3), "total_s": round(s.total_s, 3), "gen_tps": round(s.gen_tps, 2),
                     "prompt_tokens": s.prompt_tokens, "completion_tokens": s.completion_tokens,
                     "finish_reason": s.finish_reason}
                    for s in stats
                ],
            }

        short = await scenario("Sinh ngắn", lambda i: _SHORT, max_tokens)
        longctx = await scenario("Ngữ cảnh dài", _long_prompt, 64)
        longgen = await scenario("Sinh dài", lambda i: _LONG_GEN, max_tokens * 3)
        scenarios = [short, longctx, longgen]

        # --- song song -------------------------------------------------------
        par_tps: float | None = None
        if conc > 1:
            mark = ctx.gpu.mark()
            t = time.perf_counter()
            res = await asyncio.gather(
                *[ctx.lm.chat(model, _user(f"({i}) {_SHORT}"), max_tokens=max_tokens) for i in range(conc)]
            )
            wall = time.perf_counter() - t
            tokens = sum(s.completion_tokens for s in res)
            par_tps = tokens / wall
            ctx.log(f"Song song ×{conc}: tổng {par_tps:.1f} tok/s ({tokens} token trong {wall:.1f} giây)")
            scenarios.append(
                {"title": f"Song song ×{conc}", "prompt_tokens": res[0].prompt_tokens, "completion_tokens": tokens,
                 "tokens_total": tokens,
                 "ttft_ms": _med([s.ttft_s * 1000 for s in res]), "gen_tps": par_tps, "prompt_tps": 0.0,
                 "gpu": ctx.gpu.summary(mark), "runs": []}
            )
            step()

        # --- tổng hợp --------------------------------------------------------
        parts = [short["gen_tps"], longgen["gen_tps"], longctx["prompt_tps"] / 10]
        if par_tps:
            parts.append(par_tps)
        parts = [x for x in parts if x > 0]
        score = round(10 * math.exp(sum(math.log(x) for x in parts) / len(parts))) if parts else 0

        overall = ctx.gpu.summary(start)
        power = longgen["gpu"]["power_avg_w"]
        summary = [
            {"label": "Điểm tổng", "value": score, "unit": "điểm"},
            {"label": "Tốc độ sinh", "value": _r(longgen["gen_tps"]), "unit": "tok/s"},
            {"label": "Độ trễ token đầu", "value": _r(short["ttft_ms"], 0), "unit": "ms"},
            {"label": "Tốc độ đọc prompt", "value": _r(longctx["prompt_tps"], 0), "unit": "tok/s"},
        ]
        if par_tps:
            summary.append({"label": f"Song song ×{conc}", "value": _r(par_tps), "unit": "tok/s"})
        if ctx.gpu.available:
            summary += [
                {"label": "VRAM đỉnh", "value": overall["vram_peak_mb"], "unit": "MB"},
                {"label": "GPU sử dụng (sinh dài)", "value": longgen["gpu"]["util_avg"], "unit": "%"},
                {"label": "Nhiệt độ cao nhất", "value": overall["temp_max"], "unit": "°C"},
                {"label": "Công suất TB (sinh dài)", "value": power, "unit": "W"},
            ]
            if power:
                summary.append({"label": "Hiệu suất", "value": _r(longgen["gen_tps"] / power, 2), "unit": "tok/s/W"})
            wh = longgen["gpu"]["energy_wh"]
            if wh and longgen["tokens_total"]:
                summary.append({"label": "Điện mỗi 1.000 token (sinh dài)", "unit": "Wh",
                                "value": _r(wh / longgen["tokens_total"] * 1000, 3)})
            if overall["energy_wh"] is not None:
                summary.append({"label": "Điện GPU tiêu thụ (phần đo)", "value": _r(overall["energy_wh"], 2), "unit": "Wh"})
        if load_s is not None:
            summary.append({"label": "Thời gian nạp model", "value": _r(load_s), "unit": "giây"})

        table = {
            "title": "Chi tiết từng kịch bản (trung vị)",
            "columns": ["Kịch bản", "Token vào", "Token ra", "TTFT (ms)", "Sinh (tok/s)", "Đọc prompt (tok/s)",
                        "GPU TB (%)", "VRAM đỉnh (MB)", "Công suất TB (W)", "Điện (Wh)"],
            "rows": [
                [s["title"], s["prompt_tokens"], s["completion_tokens"], _r(s["ttft_ms"], 0), _r(s["gen_tps"]),
                 _r(s["prompt_tps"], 0) if s is longctx else "-", s["gpu"]["util_avg"], s["gpu"]["vram_peak_mb"], s["gpu"]["power_avg_w"],
                 s["gpu"]["energy_wh"]]
                for s in scenarios
            ],
        }
        return {
            "summary": summary,
            "tables": [table],
            "charts": [ctx.gpu.chart()] if ctx.gpu.available else [],
            "raw": {"model_info": info, "load_time_s": _r(load_s, 2), "scenarios": scenarios, "gpu_overall": overall},
        }


MODULE = GpuBenchmark()

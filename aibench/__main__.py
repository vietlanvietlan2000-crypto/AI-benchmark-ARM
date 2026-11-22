"""Dòng lệnh.

  python -m aibench serve                       # mở giao diện web
  python -m aibench models                      # liệt kê model Ollama
  python -m aibench modules                     # liệt kê module và tham số
  python -m aibench run gpu_benchmark -p model=gpt-oss:20b -p runs=3
"""
from __future__ import annotations

import argparse
import asyncio
import sys

from .core import LM_URL, modules, run_module
from .ollama import Ollama


async def _models() -> None:
    lm = Ollama(LM_URL)
    try:
        if not await lm.is_up():
            sys.exit(f"Không kết nối được Ollama tại {LM_URL}. Mở ứng dụng Ollama hoặc chạy: ollama serve")
        for m in await lm.list_models():
            print(f"{m.get('type', '?'):11} {m.get('state', ''):11} {(m.get('size_bytes') or 0) / 2**30:6.1f} GB  {m['id']}")
    finally:
        await lm.close()


async def _run(module_id: str, pairs: list[str]) -> None:
    params = dict(p.split("=", 1) for p in pairs)
    rec = await run_module(module_id, params, log=print)
    print(f"\n=== {rec['module_name']} | {rec['gpu'].get('name', 'GPU ?')} | {rec['duration_s']} giây ===")
    for s in rec["result"].get("summary", []):
        print(f"  {s['label']:28} {s['value']} {s.get('unit', '')}")
    for t in rec["result"].get("tables", []):
        print(f"\n{t['title']}")
        for row in t["rows"]:
            print("  " + " | ".join(str(c) for c in row))
    for f in rec["result"].get("files", []):
        print(f"\n{f['label']}: {f['path']}")
    print(f"\nĐã lưu: results/{rec['id']}.json")


def main() -> None:
    for stream in (sys.stdout, sys.stderr):  # console Windows mặc định không phải UTF-8
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(prog="aibench", description="AI Bench Spark: benchmark và chatbot AI chạy qua Ollama (Windows on ARM)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="Mở giao diện web")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8770)  # khác cổng 8765 của AI Bench để chạy song song được
    sub.add_parser("models", help="Liệt kê model Ollama")
    sub.add_parser("modules", help="Liệt kê module")
    st = sub.add_parser("setup", help="Kiểm tra máy, cài phần mềm còn thiếu, gợi ý và tải model")
    st.add_argument("-y", "--yes", action="store_true", help="Không hỏi, lấy lựa chọn mặc định")
    st.add_argument("--lang", choices=["vi", "en"])
    r = sub.add_parser("run", help="Chạy 1 module từ dòng lệnh")
    r.add_argument("module")
    r.add_argument("-p", "--param", action="append", default=[], metavar="KEY=VALUE")
    a = ap.parse_args()

    if a.cmd == "serve":
        import uvicorn

        print(f"AI Bench Spark: http://{a.host}:{a.port}")
        uvicorn.run("aibench.server:app", host=a.host, port=a.port, log_level="warning")
    elif a.cmd == "setup":
        from .setup import main as setup_main

        setup_main(a.yes, a.lang)
    elif a.cmd == "models":
        asyncio.run(_models())
    elif a.cmd == "modules":
        for m in modules().values():
            print(f"{m.id}: {m.name}\n  {m.description}")
            for p in m.params:
                print(f"    -p {p.key}=<{p.type}>  {p.label} (mặc định: {p.default})")
    elif a.cmd == "run":
        try:
            asyncio.run(_run(a.module, a.param))
        except (RuntimeError, ValueError) as e:
            sys.exit(f"Lỗi: {e}")


if __name__ == "__main__":
    main()

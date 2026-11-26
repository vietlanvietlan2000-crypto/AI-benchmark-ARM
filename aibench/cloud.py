"""Lưu kết quả benchmark lên Google Drive qua một Apps Script trung gian (aibench/cloud_script.gs).

Kết quả vẫn nằm ở thư mục results/ trên máy; nút "Đẩy lên cloud" chép các file đã chọn lên thư mục Google Drive,
nút "Lấy từ cloud" chép về các file máy chưa có. Cấu hình (link script + khóa) lưu trong cloud.json, không đưa lên GitHub.
"""
from __future__ import annotations

import asyncio
import json
import re
import secrets
from pathlib import Path
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse

from .core import RESULTS_DIR, ROOT

router = APIRouter(prefix="/api/cloud")

FOLDER_ID = "1Hk707a8uEyz6kxM5aNQlfnrJpHhi6F-q"
FOLDER_URL = f"https://drive.google.com/drive/folders/{FOLDER_ID}"
CONFIG = ROOT / "cloud.json"
SCRIPT = Path(__file__).with_name("cloud_script.gs")
NAME = re.compile(r"[\w\-]+\.json")

state: dict[str, Any] = {"running": False, "op": "", "done": 0, "total": 0, "file": "", "error": None,
                         "sent": 0, "skipped": 0, "got": 0}


def config() -> dict[str, str]:
    try:
        c = json.loads(CONFIG.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        c = {}
    if not c.get("key"):  # khóa bí mật ghép vào script; tạo một lần cho máy này
        c["key"] = secrets.token_urlsafe(24)
        save(c)
    return {"url": str(c.get("url") or ""), "key": c["key"]}


def save(c: dict[str, str]) -> None:
    CONFIG.write_text(json.dumps(c, ensure_ascii=False, indent=2), encoding="utf-8")


async def call(action: str, **kw: Any) -> dict[str, Any]:
    c = config()
    if not c["url"]:
        raise RuntimeError("Chưa nhập link Apps Script. Bấm \"Cài đặt cloud\" để cài.")
    # Apps Script trả lời qua một lần chuyển hướng (302) sang googleusercontent.com
    async with httpx.AsyncClient(timeout=httpx.Timeout(300, connect=15), follow_redirects=True) as http:
        r = await http.post(c["url"], json={"key": c["key"], "action": action, **kw})
    try:
        data = r.json()
    except ValueError:
        if "accounts.google.com" in str(r.url) or "<html" in r.text[:200].lower():
            raise RuntimeError("Google đòi đăng nhập: khi triển khai script phải chọn \"Người có quyền truy cập: Bất kỳ ai\".") from None
        raise RuntimeError(f"Google trả lời không đúng dạng ({r.status_code}): {r.text[:200]}") from None
    if not data.get("ok"):
        raise RuntimeError(str(data.get("error") or "Lỗi không rõ từ Apps Script"))
    return data


async def remote_files() -> dict[str, int]:
    return {f["name"]: int(f.get("size") or 0) for f in (await call("list"))["files"] if NAME.fullmatch(f["name"])}


@router.get("")
async def info() -> dict[str, Any]:
    c = config()
    return {"configured": bool(c["url"]), "url": c["url"], "folder_url": FOLDER_URL, "state": state}


@router.get("/script", response_class=PlainTextResponse)
async def script() -> str:
    """Script đã điền sẵn thư mục Google Drive và khóa của máy này, để dán vào script.google.com."""
    c = config()
    return SCRIPT.read_text(encoding="utf-8").replace("__FOLDER_ID__", FOLDER_ID).replace("__KEY__", c["key"])


@router.put("")
async def set_url(body: dict[str, Any]) -> dict[str, Any]:
    url = str(body.get("url") or "").strip()
    if url and not re.fullmatch(r"https://script\.google\.com/macros/s/[\w\-]+/exec", url):
        raise HTTPException(400, "Link phải có dạng https://script.google.com/macros/s/…/exec (URL ứng dụng web)")
    c = config()
    c["url"] = url
    save(c)
    if not url:
        return {"ok": True}
    try:
        folder = (await call("ping")).get("folder", "")
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"Đã lưu link nhưng chưa kết nối được: {e}") from e
    return {"ok": True, "folder": folder}


@router.get("/files")
async def files() -> dict[str, Any]:
    try:
        return {"files": await remote_files()}
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, str(e)) from e


def _start(op: str, work: Any) -> dict[str, Any]:
    if state["running"]:
        raise HTTPException(409, "Đang đồng bộ cloud, đợi xong đã")
    if not config()["url"]:
        raise HTTPException(400, "Chưa nhập link Apps Script. Bấm \"Cài đặt cloud\" để cài.")
    state.update(running=True, op=op, done=0, total=0, file="", error=None, sent=0, skipped=0, got=0)

    async def run() -> None:
        try:
            await work()
        except Exception as e:  # noqa: BLE001
            state["error"] = str(e) or type(e).__name__
        finally:
            state.update(running=False, file="")

    asyncio.get_running_loop().create_task(run())
    return {"ok": True}


@router.post("/push")
async def push(body: dict[str, Any]) -> dict[str, Any]:
    ids = [str(i) for i in body.get("ids") or []]
    paths = [RESULTS_DIR / f"{i}.json" for i in ids if re.fullmatch(r"[\w\-]+", i)]
    paths = [p for p in paths if p.is_file()]
    if not paths:
        raise HTTPException(400, "Chưa chọn kết quả nào")

    async def work() -> None:
        state["total"] = len(paths)
        have = await remote_files()
        for p in paths:
            state["file"] = p.name
            if have.get(p.name) == p.stat().st_size:  # đã có trên cloud, giống hệt
                state["skipped"] += 1
            else:
                await call("put", name=p.name, content=p.read_bytes().decode("utf-8"))  # gửi nguyên từng byte để so dung lượng được
                state["sent"] += 1
            state["done"] += 1

    return _start("push", work)


@router.post("/pull")
async def pull() -> dict[str, Any]:
    async def work() -> None:
        have = await remote_files()
        todo = [n for n in sorted(have) if not (RESULTS_DIR / n).exists()]  # không ghi đè kết quả trên máy
        state["total"], state["skipped"] = len(todo), len(have) - len(todo)
        RESULTS_DIR.mkdir(exist_ok=True)
        for n in todo:
            state["file"] = n
            content = (await call("get", name=n))["content"]
            try:
                json.loads(content)
            except ValueError:
                raise RuntimeError(f"File {n} trên cloud không phải JSON hợp lệ") from None
            tmp = RESULTS_DIR / (n + ".part")
            tmp.write_bytes(content.encode("utf-8"))
            tmp.replace(RESULTS_DIR / n)
            state["got"] += 1
            state["done"] += 1

    return _start("pull", work)

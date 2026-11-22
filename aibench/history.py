"""Lịch sử hội thoại của chatbot: lưu trong SQLite chat_history/chats.db (dữ liệu riêng của từng máy).

Giao diện tự lưu cả cuộc trò chuyện sau mỗi câu trả lời (gồm nguồn trích dẫn, thời gian trả lời…), nên mở lại là
xem được y như lúc chat, bấm vào trích dẫn vẫn mở được trang tài liệu, và chat tiếp được ngay.
"""
from __future__ import annotations

import json
import re
import sqlite3
import time
from typing import Any

from fastapi import APIRouter, HTTPException

from .core import ROOT

DIR = ROOT / "chat_history"
DB = DIR / "chats.db"
KEEP = ("role", "content", "sources", "stats", "chat", "docs", "web", "error", "ctx")  # bỏ trạng thái tạm lúc đang chạy
router = APIRouter(prefix="/api/chat/conversations")


def _db() -> sqlite3.Connection:
    DIR.mkdir(exist_ok=True)
    db = sqlite3.connect(DB, timeout=10)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("CREATE TABLE IF NOT EXISTS conversations (id TEXT PRIMARY KEY, title TEXT, created REAL, updated REAL, "
               "n INTEGER, messages TEXT, auto INTEGER DEFAULT 1)")  # auto = 1: tên tự đặt, 0: người dùng đã đổi tên
    if "auto" not in {r[1] for r in db.execute("PRAGMA table_info(conversations)")}:
        db.execute("ALTER TABLE conversations ADD COLUMN auto INTEGER DEFAULT 1")
    return db


def _check_id(cid: str) -> str:
    if not re.fullmatch(r"[\w-]{6,64}", cid):
        raise HTTPException(400, "Mã cuộc trò chuyện không hợp lệ")
    return cid


def _title(messages: list[dict[str, Any]]) -> str:
    """Câu hỏi thật đầu tiên (bỏ qua câu chào hỏi xã giao mà chatbot trả lời không cần tra cứu)."""
    users = [(k, m) for k, m in enumerate(messages) if m.get("role") == "user" and m.get("content")]
    real = [m for k, m in users if not (k + 1 < len(messages) and messages[k + 1].get("chat"))]
    first = (real or [m for _, m in users] or [{"content": ""}])[0]["content"]
    t = " ".join(str(first).split())
    return (t[:70] + "…") if len(t) > 70 else (t or "Cuộc trò chuyện")


@router.get("")
async def list_conversations(q: str = "") -> list[dict[str, Any]]:
    """Các cuộc trò chuyện, mới nhất trước. q: tìm trong tiêu đề và nội dung (không phân biệt hoa thường)."""
    db = _db()
    try:
        sql, args = "SELECT id, title, created, updated, n FROM conversations", []
        if q.strip():
            sql += " WHERE lower(title) LIKE ? OR lower(messages) LIKE ?"
            like = f"%{q.strip().lower()}%"
            args = [like, like]
        rows = db.execute(sql + " ORDER BY updated DESC LIMIT 500", args).fetchall()
    finally:
        db.close()
    return [{"id": i, "title": t, "created": c, "updated": u, "n": n} for i, t, c, u, n in rows]


@router.get("/{cid}")
async def get_conversation(cid: str) -> dict[str, Any]:
    db = _db()
    try:
        row = db.execute("SELECT title, created, updated, messages FROM conversations WHERE id=?", (_check_id(cid),)).fetchone()
    finally:
        db.close()
    if not row:
        raise HTTPException(404, "Không tìm thấy cuộc trò chuyện này")
    return {"id": cid, "title": row[0], "created": row[1], "updated": row[2], "messages": json.loads(row[3])}


@router.put("/{cid}")
async def save_conversation(cid: str, body: dict[str, Any]) -> dict[str, Any]:
    """Lưu (tạo mới hoặc ghi đè) cả cuộc trò chuyện. body: {messages, title?}. Tiêu đề mặc định là câu hỏi đầu tiên."""
    msgs = [{k: m[k] for k in KEEP if k in m} for m in body.get("messages") or [] if isinstance(m, dict)]
    if not msgs:
        raise HTTPException(400, "Cuộc trò chuyện trống")
    now = time.time()
    db = _db()
    try:
        old = db.execute("SELECT title, created, auto FROM conversations WHERE id=?", (_check_id(cid),)).fetchone()
        auto = 0 if body.get("title") or (old and not old[2]) else 1
        # tên tự đặt được cập nhật theo câu hỏi thật đầu tiên (vd. mở đầu bằng "hello" rồi mới hỏi); tên người dùng đặt thì giữ
        title = str(body.get("title") or (old[0] if old and not auto else "") or _title(msgs))[:200]
        with db:
            db.execute("INSERT OR REPLACE INTO conversations VALUES (?, ?, ?, ?, ?, ?, ?)",
                       (cid, title, old[1] if old else now, now, len(msgs), json.dumps(msgs, ensure_ascii=False), auto))
    finally:
        db.close()
    return {"id": cid, "title": title, "updated": now}


@router.patch("/{cid}")
async def rename_conversation(cid: str, body: dict[str, Any]) -> dict[str, Any]:
    title = " ".join(str(body.get("title") or "").split())[:200]
    if not title:
        raise HTTPException(400, "Tên cuộc trò chuyện trống")
    db = _db()
    try:
        with db:
            n = db.execute("UPDATE conversations SET title=?, auto=0 WHERE id=?", (title, _check_id(cid))).rowcount
    finally:
        db.close()
    if not n:
        raise HTTPException(404, "Không tìm thấy cuộc trò chuyện này")
    return {"id": cid, "title": title}


@router.delete("/{cid}")
async def delete_conversation(cid: str) -> dict[str, Any]:
    db = _db()
    try:
        with db:
            db.execute("DELETE FROM conversations WHERE id=?", (_check_id(cid),))
    finally:
        db.close()
    return {"ok": True}

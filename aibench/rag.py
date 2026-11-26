"""Tra cứu tài liệu (RAG) cho chatbot.

Tài liệu đặt trong thư mục Tai_lieu/ (chia thư mục con tùy ý). Lập chỉ mục gồm 5 bước:
  1. Tiếp nhận   quét thư mục, chỉ xử lý file mới / đã sửa (hoặc tất cả khi đổi model embedding)
  2. Lấy chữ     PDF có chữ, Word, Excel, txt/md/csv: đọc thẳng; trang PDF scan và file ảnh: OCR bằng model thị giác
  3. Cắt đoạn    chia đoạn, ghi đoạn nằm ở trang / mục / sheet nào
  4. Vector hóa  model embedding của Ollama đổi mỗi đoạn thành vector
  5. Lưu trữ     ghi vào cơ sở dữ liệu SQLite rag_index/index.db (từng tài liệu một, an toàn khi bị ngắt giữa chừng)

Kết quả OCR được lưu riêng theo nội dung file, nên đổi model embedding hay lập lại chỉ mục không phải OCR lại.
Khi hỏi: tìm các đoạn gần câu hỏi nhất để biết tài liệu nào liên quan, rồi đưa cho AI đọc cả tài liệu đó
nếu vừa ngữ cảnh, hoặc các đoạn liên quan nếu tài liệu quá dài.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import re
import sqlite3
import threading
import zipfile
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from .core import LM_URL, ROOT

DOCS = ROOT / "Tai_lieu"
INDEX = ROOT / "rag_index"
DB = INDEX / "index.db"
TEXT_EXTS = {".txt", ".md", ".csv", ".pdf", ".docx", ".xlsx"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
EXTS = TEXT_EXTS | IMAGE_EXTS
CHUNK = 900  # ký tự mỗi đoạn
VERSION = 3  # đổi cách cắt đoạn / cấu trúc dữ liệu thì tăng số này để lập lại chỉ mục
MIN_PAGE_TEXT = 30  # trang PDF có ít hơn số ký tự này coi là trang scan, cần OCR
OCR_SIDE = 1600  # cạnh dài nhất (px) của ảnh gửi đi OCR
OCR_PARALLEL = 2  # số trang OCR cùng lúc (Ollama xử lý song song theo OLLAMA_NUM_PARALLEL)
OCR_MAX_TOKENS = 6000  # chữ đọc ra tối đa mỗi trang (trang dày nhất đã gặp ~4.200 token)
OCR_PROMPT = ("Chép lại toàn bộ chữ trong ảnh theo đúng thứ tự đọc. Bảng thì viết thành bảng Markdown. "
              "Chỉ trả về nội dung, không giải thích.")

Loc = list | None  # ["page", 3] | ["sheet", "Bảng giá"] | ["sec", "2. Thủ tục xin nghỉ"] | None


# ----- bước 2: lấy chữ ----------------------------------------------------------------------------
def _unxml(s: str) -> str:
    return s.replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"').replace("&apos;", "'").replace("&amp;", "&")


# pdfium không an toàn khi nhiều luồng dùng cùng lúc (lập chỉ mục + khung xem trang gốc): mọi lệnh pdfium đi qua khóa này.
PDFIUM_LOCK = threading.RLock()


def _jpeg(img: Any) -> bytes:
    img = img.convert("RGB")
    img.thumbnail((OCR_SIDE, OCR_SIDE))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=88)
    return buf.getvalue()


def read_parts(path: Path) -> list[dict[str, Any]]:
    """Các phần của tài liệu kèm vị trí: {loc, text} hoặc {loc, image} nếu phần đó cần OCR.
    Mỗi trang PDF, mỗi sheet Excel, mỗi mục (theo tiêu đề) Word/Markdown là một phần; file ảnh là một phần cần OCR."""
    ext = path.suffix.lower()
    if ext in IMAGE_EXTS:
        from PIL import Image

        with Image.open(path) as im:
            return [{"loc": None, "page": 1, "image": _jpeg(im)}]
    if ext == ".pdf":
        from pypdf import PdfReader

        out = []
        scans = []
        for n, pg in enumerate(PdfReader(str(path)).pages, 1):
            text = pg.extract_text() or ""
            if len(text.strip()) >= MIN_PAGE_TEXT:
                out.append({"loc": ["page", n], "text": text})
            else:
                out.append({"loc": ["page", n], "page": n})
                scans.append(n)
        if scans:  # trang không có lớp chữ → vẽ thành ảnh để OCR
            import pypdfium2 as pdfium

            for part in out:
                if "page" in part:
                    with PDFIUM_LOCK:  # mở lại mỗi trang để luồng xem trang gốc không phải chờ cả tài liệu lớn
                        pdf = pdfium.PdfDocument(str(path))
                        try:
                            pg = pdf[part["page"] - 1]
                            w, h = pg.get_size()
                            img = pg.render(scale=min(3.0, OCR_SIDE / max(w, h, 1))).to_pil()
                        finally:
                            pdf.close()
                    part["image"] = _jpeg(img)
        return out
    if ext == ".docx":
        with zipfile.ZipFile(path) as z:
            xml = z.read("word/document.xml").decode("utf-8", "replace")
        out = [{"loc": None, "text": ""}]
        for p in re.findall(r"<w:p[ >].*?</w:p>", xml, flags=re.S):
            text = _unxml("".join(re.findall(r"<w:t[^>]*>([^<]*)</w:t>", p))).strip()
            if not text:
                continue
            # Kiểu tiêu đề: "Heading1" (Word tiếng Anh), "u1" (Word tiếng Việt), "Title"
            if re.search(r'<w:pStyle w:val="(Heading\d|u\d|Title)"', p) and len(text) < 200:
                out.append({"loc": ["sec", text], "text": text})
            else:
                out[-1]["text"] = (out[-1]["text"] + "\n" + text).strip()
        return out
    if ext == ".xlsx":
        from openpyxl import load_workbook

        wb = load_workbook(path, read_only=True, data_only=True)
        out = []
        for ws in wb.worksheets:
            rows = [" | ".join(str(c) for c in row if c is not None) for row in ws.iter_rows(values_only=True)]
            out.append({"loc": ["sheet", ws.title], "text": "\n".join(r for r in rows if r)})
        return out
    raw = path.read_bytes()
    text = raw.decode("utf-8", "replace")
    for enc in ("utf-8-sig", "utf-16", "cp1258"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeError:
            continue
    out = [{"loc": None, "text": ""}]
    for line in text.replace("\r", "").split("\n"):  # tiêu đề Markdown (# ...) mở một mục mới
        m = re.match(r"#{1,6}\s+(.+)", line)
        if m:
            out.append({"loc": ["sec", m.group(1).strip()], "text": line})
        else:
            out[-1]["text"] += "\n" + line
    return out


def ocr_to_text(s: str) -> str:
    """Model OCR chuyên dụng (vd. chandra-ocr) trả về HTML có toạ độ; đổi thành chữ thường, bảng thành dòng "a | b"."""
    if not re.search(r"<(div|table|p)\b", s):
        return s.strip()
    import lxml.html

    root = lxml.html.fragment_fromstring(s, create_parent="div")
    for br in root.iter("br"):  # <br/> trong ô bảng: giữ khoảng trắng giữa hai dòng chữ
        br.tail = " " + (br.tail or "")
    lines: list[str] = []
    for block in root:
        if block.get("data-label") in ("Image", "Figure", "Picture"):
            continue
        tables = block.xpath(".//table") if block.tag != "table" else [block]
        if tables:
            for tb in tables:
                for tr in tb.iter("tr"):
                    cells = [" ".join(td.text_content().split()) for td in tr if td.tag in ("td", "th")]
                    if any(cells):
                        lines.append("| " + " | ".join(cells) + " |")
            lines.append("")
        else:
            t = " ".join(block.text_content().split())
            if t:
                lines.append(t)
    return "\n".join(lines).strip()


def _cut_loop(s: str) -> str:
    """Model OCR đôi khi lặp mãi một mẩu (hay gặp nhất: một hàng bảng trống) tới khi hết số token.
    Giữ tối đa 3 lần lặp liền nhau của cùng một khối dòng, hoặc của cùng một mẩu chữ trong một dòng."""
    # lặp trong một dòng, vd. "| 产品批号 | 产品批号 | 产品批号 | …"
    lines = [re.sub(r"(.{4,200}?)\1{3,}", r"\1\1\1", ln) for ln in s.split("\n")]
    for size in range(1, 41):  # khối lặp dài 1–40 dòng
        out: list[str] = []
        i, cut = 0, False
        while i < len(lines):
            block = lines[i:i + size]
            reps = 1
            while lines[i + reps * size:i + (reps + 1) * size] == block and len(block) == size:
                reps += 1
            if reps > 3 and any(x.strip() for x in block):
                out += block * 3
                i += reps * size
                cut = True
            else:
                out.append(lines[i])
                i += 1
        if cut:
            lines = out
    return "\n".join(lines)


async def ocr_image(model: str, jpeg: bytes) -> str:
    from .ollama import KEEP_ALIVE, can_think, ctx_of

    payload = {"model": model, "stream": False, "keep_alive": KEEP_ALIVE,
               "options": {"temperature": 0, "num_predict": OCR_MAX_TOKENS, "num_ctx": ctx_of(model)},
               "messages": [{"role": "user", "content": OCR_PROMPT, "images": [base64.b64encode(jpeg).decode()]}]}
    if await can_think(model):
        payload["think"] = False  # chép chữ không cần suy nghĩ
    async with httpx.AsyncClient(timeout=httpx.Timeout(900, connect=5)) as http:
        r = await http.post(f"{LM_URL}/api/chat", json=payload)
    if r.status_code != 200:
        raise RuntimeError(f"Ollama trả lỗi {r.status_code} khi OCR: {r.text[:300]}")
    data = r.json()
    msg = data.get("message") or {}
    if not (msg.get("content") or "").strip() and msg.get("thinking"):
        # Bản "thinking" (vd. tag mặc định qwen3-vl:2b) bỏ qua think=false: chỉ suy nghĩ, không chép chữ ra.
        raise RuntimeError(f"Model OCR {model} chỉ suy nghĩ mà không chép chữ ra. Hãy dùng bản -instruct, "
                           f"vd. ollama pull qwen3-vl:2b-instruct (hoặc qwen3-vl:8b-instruct)")
    text = msg.get("content") or ""  # bản gốc model trả về; ocr_to_text() khi dùng
    return _cut_loop(text) if data.get("done_reason") == "length" else text


# ----- bước 3: cắt đoạn ---------------------------------------------------------------------------
def split(text: str) -> list[str]:
    """Cắt theo đoạn văn, gom lại cho gần CHUNK ký tự."""
    parts: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        while len(para) > CHUNK * 1.5:  # đoạn quá dài (bảng, PDF không xuống dòng) thì cắt ở chỗ xuống dòng / khoảng trắng
            cut = max(para.rfind("\n", 0, CHUNK), para.rfind(" ", 0, CHUNK))
            cut = cut if cut > CHUNK // 2 else CHUNK
            parts.append(para[:cut])
            para = para[cut:].strip()
        if para:
            parts.append(para)
    out: list[str] = []
    for p in parts:
        if out and (len(out[-1]) + len(p) < CHUNK or len(p) < 40):  # mẩu quá ngắn (chữ ký, số trang) gộp vào đoạn trước
            out[-1] += "\n\n" + p
        else:
            out.append(p)
    return out


def loc_text(loc: Loc) -> str:
    """Vị trí viết cho AI đọc (tiếng Việt)."""
    if not loc:
        return ""
    return {"page": "trang", "sheet": "sheet", "sec": "mục"}[loc[0]] + f" {loc[1]}"


def list_files() -> list[Path]:
    if not DOCS.is_dir():
        return []
    return sorted(p for p in DOCS.rglob("*") if p.is_file() and p.suffix.lower() in EXTS and not p.name.startswith("~$"))


def rel(p: Path) -> str:
    return p.relative_to(DOCS).as_posix()


def sha1(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ----- bước 4: embedding --------------------------------------------------------------------------
def _prefix(model: str, query: bool) -> str:
    """Một số model embedding cần câu dẫn riêng cho câu hỏi và cho tài liệu."""
    m = model.lower()
    if "nomic" in m:
        return "search_query: " if query else "search_document: "
    if ("qwen3-embedding" in m or "qwen3-vl-embedding" in m) and query:
        return "Instruct: Given a question, retrieve passages from internal documents that answer it\nQuery: "
    if "e5" in m:
        return "query: " if query else "passage: "
    return ""


async def embed(model: str, texts: list[str], query: bool = False) -> np.ndarray:
    pre = _prefix(model, query)
    from .ollama import KEEP_ALIVE, ctx_of

    payload = {"model": model, "input": [pre + t for t in texts], "truncate": True, "keep_alive": KEEP_ALIVE,
               "options": {"num_ctx": ctx_of(model)}}
    async with httpx.AsyncClient(timeout=httpx.Timeout(600, connect=5)) as http:
        for attempt in (1, 2):  # lần đầu Ollama phải nạp model embedding, đôi khi lỗi thoáng qua
            r = await http.post(f"{LM_URL}/api/embed", json=payload)
            if r.status_code == 200 or attempt == 2:
                break
            await asyncio.sleep(3)
    if r.status_code != 200:
        hint = " Có thể bộ nhớ đã đầy: hãy gỡ bớt model đang nạp hoặc chọn model embedding nhỏ hơn." if "memory" in r.text.lower() else ""
        raise RuntimeError(f"Ollama trả lỗi {r.status_code} khi tạo embedding: {r.text[:300]}{hint}")
    v = np.array(r.json().get("embeddings") or [], dtype=np.float32)
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)


# ----- bước 5: cơ sở dữ liệu ----------------------------------------------------------------------
SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS docs (path TEXT PRIMARY KEY, mtime REAL, size INTEGER, sha1 TEXT,
                                 n INTEGER, note TEXT, ocr_pages INTEGER, ocr_missing INTEGER);
CREATE TABLE IF NOT EXISTS chunks (doc TEXT, i INTEGER, loc TEXT, text TEXT, ocr INTEGER, vec BLOB, PRIMARY KEY (doc, i));
CREATE TABLE IF NOT EXISTS ocr (sha1 TEXT, page INTEGER, model TEXT, text TEXT, PRIMARY KEY (sha1, page, model));
"""


def connect() -> sqlite3.Connection:
    INDEX.mkdir(exist_ok=True)
    db = sqlite3.connect(DB, timeout=30)
    db.execute("PRAGMA journal_mode=WAL")
    db.executescript(SCHEMA)
    if (db.execute("SELECT value FROM meta WHERE key='version'").fetchone() or [None])[0] != str(VERSION):
        with db:  # chỉ mục kiểu cũ: bỏ chỉ mục, giữ kết quả OCR
            db.execute("DELETE FROM chunks")
            db.execute("DELETE FROM docs")
            db.execute("INSERT OR REPLACE INTO meta VALUES ('version', ?)", (str(VERSION),))
            db.execute("DELETE FROM meta WHERE key='model'")
    return db


class Index:
    """Bản sao trong bộ nhớ của chỉ mục (để tìm nhanh); nguồn gốc là SQLite."""

    def __init__(self) -> None:
        self.model = ""
        self.files: dict[str, dict[str, Any]] = {}  # rel -> {mtime, size, n, note, ocr_pages, ocr_missing}
        self.chunks: list[dict[str, Any]] = []  # {doc, i, loc, text, ocr}
        self.vecs = np.zeros((0, 1), dtype=np.float32)

    @classmethod
    def load(cls) -> "Index":
        ix = cls()
        db = connect()
        try:
            ix.model = (db.execute("SELECT value FROM meta WHERE key='model'").fetchone() or [""])[0]
            for path, mtime, size, n, note, op, om in db.execute(
                    "SELECT path, mtime, size, n, note, ocr_pages, ocr_missing FROM docs"):
                ix.files[path] = {"mtime": mtime, "size": size, "n": n, "note": note or "", "ocr_pages": op or 0, "ocr_missing": om or 0}
            vecs = []
            for doc, i, loc, text, ocr, vec in db.execute("SELECT doc, i, loc, text, ocr, vec FROM chunks ORDER BY doc, i"):
                ix.chunks.append({"doc": doc, "i": i, "loc": json.loads(loc) if loc else None, "text": text, "ocr": bool(ocr)})
                vecs.append(np.frombuffer(vec, dtype=np.float32))
            if vecs:
                ix.vecs = np.stack(vecs)
        finally:
            db.close()
        return ix

    def stale(self) -> list[str]:
        """Các file mới hoặc đã đổi so với chỉ mục."""
        out = []
        for p in list_files():
            f, s = self.files.get(rel(p)), p.stat()
            if not f or f["mtime"] != s.st_mtime or f["size"] != s.st_size:
                out.append(rel(p))
        return out

    def of(self, doc: str) -> list[dict[str, Any]]:
        return [c for c in self.chunks if c["doc"] == doc]


class Cancelled(Exception):
    pass


async def build(embed_model: str, ocr_model: str, progress: dict[str, Any],
                prepare_ocr: Callable[[str], Awaitable[None]], prepare_embed: Callable[[str], Awaitable[None]],
                only: str = "") -> Index:
    """Lập chỉ mục 5 bước. `progress` được cập nhật liên tục cho giao diện; progress["cancel"] = True để dừng
    (phần đã làm được giữ lại: trang đã OCR, tài liệu đã lưu).
    `only` = đường dẫn (tương đối) một tài liệu: chỉ làm lại tài liệu đó từ đầu: đọc lại, OCR lại (bỏ kết quả OCR
    đã lưu của model OCR đang chọn), vector hóa và lưu lại; các tài liệu khác giữ nguyên."""
    def check() -> None:
        if progress.get("cancel"):
            raise Cancelled

    db = connect()
    try:
        # ---- bước 1: tiếp nhận
        progress.update(stage="scan", done=0, total=0, file="")
        old = Index.load()
        files = list_files()
        names = {rel(p) for p in files}
        if only:
            if only not in names:
                raise RuntimeError(f"Không tìm thấy tài liệu {only} trong thư mục Tai_lieu")
            if old.model and old.model != embed_model:
                raise RuntimeError(f"Chỉ mục đang dùng model {old.model}. Đổi model lập chỉ mục thì phải bấm "
                                   "\"Cập nhật chỉ mục\" cho toàn bộ tài liệu, không làm lại riêng một tài liệu được.")
        with db:
            for gone in set(old.files) - names:  # file đã bị xoá khỏi thư mục
                db.execute("DELETE FROM docs WHERE path=?", (gone,))
                db.execute("DELETE FROM chunks WHERE doc=?", (gone,))
            if old.model != embed_model and not only:  # đổi model embedding: mọi vector cũ không dùng được nữa
                db.execute("DELETE FROM docs")
                db.execute("DELETE FROM chunks")
                db.execute("INSERT OR REPLACE INTO meta VALUES ('model', ?)", (embed_model,))
            if only and not old.model:
                db.execute("INSERT OR REPLACE INTO meta VALUES ('model', ?)", (embed_model,))
        todo = [p for p in files if rel(p) == only] if only else files if old.model != embed_model else [p for p in files if rel(p) in set(old.stale())]
        # Tài liệu có trang scan chưa OCR được (lần trước chưa chọn model OCR) thì làm lại khi đã có model OCR.
        if ocr_model and not only:
            todo += [p for p in files if p not in todo and (old.files.get(rel(p)) or {}).get("ocr_missing")]
        # Tài liệu lần trước đọc lỗi thì lần này đọc lại (file không đổi nên bước trên không tự chọn lại).
        todo += [] if only else [p for p in files if p not in todo and str((old.files.get(rel(p)) or {}).get("note") or "").startswith("error")]

        # ---- bước 2a: đọc chữ có sẵn, tìm các trang cần OCR
        progress.update(stage="read", done=0, total=len(todo))
        docs: list[dict[str, Any]] = []
        for p in todo:
            check()
            progress["file"] = rel(p)
            d = {"path": p, "rel": rel(p), "stat": p.stat(), "note": "", "parts": []}
            for attempt in range(2):  # lỗi tạm thời (pdfium, file đang bị chép…) thì đọc lại một lần
                try:
                    d["sha1"] = await asyncio.to_thread(sha1, p)
                    d["parts"] = await asyncio.to_thread(read_parts, p)
                    d["note"] = ""
                    if only and d["sha1"]:  # làm lại từ đầu: bỏ kết quả OCR và vị trí chữ đã lưu của tài liệu này
                        with db:
                            if ocr_model:
                                db.execute("DELETE FROM ocr WHERE sha1=? AND model=?", (d["sha1"], ocr_model))
                            try:
                                db.execute("DELETE FROM layout WHERE sha1=?", (d["sha1"],))
                            except sqlite3.OperationalError:
                                pass  # chưa có bảng layout (chưa xem trang scan nào)
                    break
                except Exception as e:  # noqa: BLE001
                    d["note"], d["sha1"], d["parts"] = f"error: {e}", "", []
            docs.append(d)
            progress["done"] += 1

        # ---- bước 2b: OCR (dùng lại kết quả đã lưu nếu có)
        need = []
        for d in docs:
            for part in d["parts"]:
                if "image" not in part:
                    continue
                row = db.execute("SELECT text FROM ocr WHERE sha1=? AND page=? AND model=?",
                                 (d["sha1"], part["page"], ocr_model)).fetchone() if ocr_model else None
                if row:
                    part["text"], part["ocr"] = ocr_to_text(row[0]), True
                elif ocr_model:
                    need.append((d, part))
        if need:
            progress.update(stage="ocr", done=0, total=len(need), file="")
            await prepare_ocr(ocr_model)
            queue = list(need)

            async def worker() -> None:
                while queue:
                    check()
                    d, part = queue.pop(0)
                    progress["file"] = f"{d['rel']} · trang {part['page']}" if d["path"].suffix.lower() == ".pdf" else d["rel"]
                    try:
                        raw = await ocr_image(ocr_model, part["image"])
                    except BaseException:
                        queue.clear()  # một trang lỗi thì các luồng khác cũng dừng
                        raise
                    part["text"], part["ocr"] = ocr_to_text(raw), True
                    with db:  # lưu bản gốc để sau này đổi cách chuyển HTML → chữ vẫn không phải OCR lại
                        db.execute("INSERT OR REPLACE INTO ocr VALUES (?, ?, ?, ?)", (d["sha1"], part["page"], ocr_model, raw))
                    progress["done"] += 1

            await asyncio.gather(*(worker() for _ in range(OCR_PARALLEL)))

        # ---- bước 3 + 4 + 5: cắt đoạn, vector hóa, lưu từng tài liệu
        progress.update(stage="embed", done=0, total=len(docs), file="")
        if docs:
            await prepare_embed(embed_model)
        for d in docs:
            check()
            progress["file"] = d["rel"]
            chunks = []
            for part in d["parts"]:
                if part.get("text"):
                    chunks += [{"loc": part["loc"], "text": t, "ocr": bool(part.get("ocr"))} for t in split(part["text"])]
            missing = sum(1 for part in d["parts"] if "image" in part and not part.get("ocr"))
            ocr_pages = sum(1 for part in d["parts"] if part.get("ocr"))
            note = d["note"] or ("empty" if not chunks and missing else "")
            head = d["rel"].rsplit(".", 1)[0].replace("/", " > ")  # đường dẫn thư mục và tên mục giúp tìm đúng tài liệu
            texts = [f"{head}{' > ' + str(c['loc'][1]) if c['loc'] and c['loc'][0] == 'sec' else ''}\n{c['text']}" for c in chunks]
            vecs = [await embed(embed_model, texts[b:b + 16]) for b in range(0, len(texts), 16)]
            vec = np.concatenate(vecs) if vecs else np.zeros((0, 1), dtype=np.float32)
            st = d["stat"]
            with db:  # 1 tài liệu = 1 giao dịch: ngắt giữa chừng thì tài liệu này chưa có, lần sau làm lại
                db.execute("DELETE FROM chunks WHERE doc=?", (d["rel"],))
                db.executemany("INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?)", [
                    (d["rel"], i, json.dumps(c["loc"], ensure_ascii=False) if c["loc"] else None, c["text"], int(c["ocr"]),
                     vec[i].astype(np.float32).tobytes()) for i, c in enumerate(chunks)])
                db.execute("INSERT OR REPLACE INTO docs VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                           (d["rel"], st.st_mtime, st.st_size, d["sha1"], len(chunks), note, ocr_pages, missing))
            progress["done"] += 1
    finally:
        db.close()
    return Index.load()


# ----- tìm và chọn đoạn cho AI đọc --------------------------------------------------------------
def words(s: str) -> set[str]:
    return {w for w in re.findall(r"\w+", s.lower()) if len(w) > 1}


async def retrieve(ix: Index, query: str | list[str], budget: int, top_docs: int = 3) -> list[dict[str, Any]]:
    """Các đoạn cho AI đọc (theo thứ tự tài liệu, rồi thứ tự trong tài liệu). `budget` là tổng số ký tự tối đa.
    Tài liệu vừa ngữ cảnh thì lấy trọn; tài liệu dài thì lấy đoạn trúng kèm đoạn liền trước/sau.
    `query` có thể là nhiều câu (vd. câu hỏi tiếng Việt + bản tiếng Anh): mỗi câu tự xếp hạng rồi trộn thứ hạng,
    nhờ vậy tài liệu tiếng Anh (datasheet, tiêu chuẩn) không bị tài liệu tiếng Việt cùng chủ đề lấn át."""
    if not ix.chunks:
        return []
    queries = [q for q in ([query] if isinstance(query, str) else query) if q and q.strip()]
    qv = await embed(ix.model, queries, query=True)
    vecs = [ix.vecs @ v for v in qv]
    cw = [words(c["doc"] + " " + c["text"]) for c in ix.chunks]
    qws = [words(q) for q in queries]

    def keyword(idx: list[int], qw: set[str]) -> np.ndarray:
        """Điểm trùng từ khóa của các đoạn idx (0..1). Từ càng hiếm trong nhóm đoạn này càng nặng ký (IDF);
        từ của câu hỏi không có trong nhóm (vd. "theo", "bao nhiêu") thì bỏ qua."""
        df = {w: sum(1 for k in idx if w in cw[k]) for w in qw}
        idf = {w: float(np.log(1 + len(idx) / n)) for w, n in df.items() if n}
        total = sum(idf.values())
        if not total:
            return np.zeros(len(idx), dtype=np.float32)
        return np.array([sum(v for w, v in idf.items() if w in cw[k]) / total for k in idx], dtype=np.float32)

    # Tầng 1 – chọn tài liệu: vector + từ khóa trên toàn bộ chỉ mục (tìm kiếm lai, bù cho model embedding
    # yếu tiếng Việt, tên riêng, mã số). Mỗi câu tìm kiếm tự xếp hạng tài liệu (điểm = đoạn tốt nhất + chút thưởng,
    # tối đa 0,1, khi có thêm đoạn lọt top 10). Không so điểm giữa các câu (câu tiếng Việt luôn cho điểm cao hơn với
    # tài liệu tiếng Việt), mà lấy 2 tài liệu đầu của mỗi câu rồi trộn thứ hạng (reciprocal rank fusion).
    allk = list(range(len(ix.chunks)))
    fused: dict[str, float] = {}
    picked: set[str] = set()
    for v, qw in zip(vecs, qws):
        score = v + 0.3 * keyword(allk, qw)
        best: dict[str, float] = {}
        for k in allk:
            d = ix.chunks[k]["doc"]
            best[d] = max(best.get(d, -9.0), float(score[k]))
        hits: dict[str, int] = {}
        for k in np.argsort(-score)[:10]:
            d = ix.chunks[k]["doc"]
            hits[d] = hits.get(d, 0) + 1
        for d, n in hits.items():
            best[d] += min(0.1, 0.03 * (n - 1))
        order = sorted(best, key=lambda d: -best[d])
        for r, d in enumerate(order):
            fused[d] = fused.get(d, 0.0) + 1 / (2 + r)
        picked |= {d for d in order[:2] if best[d] >= best[order[0]] - 0.25}  # bỏ tài liệu kém xa tài liệu đầu
    ranked = sorted(picked, key=lambda d: -fused[d])[:top_docs]
    pos = {}
    for k, c in enumerate(ix.chunks):
        pos.setdefault(c["doc"], []).append(k)

    out: list[dict[str, Any]] = []
    left = budget
    for n, d in enumerate(ranked):
        mine = ix.of(d)
        # chia đều phần còn lại cho các tài liệu chưa xét; tài liệu ngắn dùng không hết thì phần dư sang tài liệu sau
        share = left // (len(ranked) - n)
        if sum(len(c["text"]) for c in mine) <= share:
            keep = list(range(len(mine)))
        else:
            # Tầng 2 – chọn đoạn trong tài liệu dài: từ khóa hiếm *trong tài liệu này* (vd. "Tilting" giữa hàng chục
            # trang bảng na ná nhau) được tính nặng, rồi lấy các đoạn trúng nhất kèm đoạn liền trước/sau.
            # Mỗi câu tìm kiếm xếp hạng riêng rồi trộn thứ hạng (không so điểm giữa câu tiếng Việt và tiếng Anh).
            idx = pos[d]
            local = np.zeros(len(idx))
            for v, qw in zip(vecs, qws):
                order = np.argsort(-(v[idx] + 0.5 * keyword(idx, qw)))
                local[order] += 1 / (3 + np.arange(len(idx)))
            want: list[int] = []
            for i in (ix.chunks[idx[j]]["i"] for j in np.argsort(-local)[:8]):
                for j in (i, i - 1, i + 1):
                    if 0 <= j < len(mine) and j not in want:
                        want.append(j)
            keep, used = [], 0
            for j in want:  # đoạn quá dài so với phần còn lại thì bỏ qua, thử đoạn kế
                if used + len(mine[j]["text"]) <= share:
                    keep.append(j)
                    used += len(mine[j]["text"])
            keep.sort()
        out += [mine[j] for j in keep]
        left -= sum(len(mine[j]["text"]) for j in keep)
        if left < 500:
            break
    return out

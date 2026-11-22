"""Xem trang gốc của tài liệu và tô vàng chỗ AI đã dùng.

- vẽ trang PDF / file ảnh thành ảnh JPEG để giao diện hiện đúng như bản gốc
- tìm các dòng trên trang khớp với câu trả lời, trả về khung (tọa độ 0..1 theo bề rộng / cao trang):
    * trang PDF có chữ: lấy vị trí chữ từ PDF (pdfium), gom thành từng dòng
    * trang scan / file ảnh: RapidOCR dò vị trí từng ô chữ trên ảnh (dùng được với mọi model OCR); không có thì dùng
      tọa độ khối chữ mà model OCR (vd. chandra-ocr) trả về, chia khối thành dòng / hàng bảng
"""
from __future__ import annotations

import io
import json
import re
import sqlite3
import threading
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any

from . import rag

SIDE = 1500  # cạnh dài (px) của ảnh trang gửi lên giao diện
WORD = re.compile(r"[^\W_]+", re.U)


def resolve(rel_path: str) -> Path:
    p = (rag.DOCS / rel_path).resolve()
    if not p.is_file() or not p.is_relative_to(rag.DOCS.resolve()):
        raise FileNotFoundError(rel_path)
    return p


def kind(p: Path) -> str:
    ext = p.suffix.lower()
    return "pdf" if ext == ".pdf" else "image" if ext in rag.IMAGE_EXTS else ""


def page_count(p: Path) -> int:
    k = kind(p)
    if k == "image":
        return 1
    if k != "pdf":
        return 0
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(p))
    try:
        return len(pdf)
    finally:
        pdf.close()


@lru_cache(maxsize=24)
def _render(path: str, mtime: float, page: int) -> bytes:
    from PIL import Image

    p = Path(path)
    if kind(p) == "image":
        with Image.open(p) as im:
            img = im.convert("RGB")
    else:
        import pypdfium2 as pdfium

        pdf = pdfium.PdfDocument(path)
        try:
            pg = pdf[page - 1]
            w, h = pg.get_size()
            img = pg.render(scale=min(3.0, SIDE / max(w, h, 1))).to_pil().convert("RGB")
        finally:
            pdf.close()
    img.thumbnail((SIDE, SIDE))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=85)
    return buf.getvalue()


def render(p: Path, page: int) -> bytes:
    return _render(str(p), p.stat().st_mtime, page)


# ----- tìm dòng khớp câu trả lời -------------------------------------------------------------------
def _words(s: str) -> list[str]:
    return WORD.findall(s.lower())


def _score(line: str, answer: set[str]) -> float:
    """Như cách tô dòng ở khung "Văn bản": phần lớn dòng nằm trong câu trả lời, hoặc phần lớn câu trả lời nằm trong dòng."""
    ws = _words(line)
    if len(ws) < 3 or line.lstrip().startswith("#"):
        return 0.0
    lw = set(ws)
    in_ans = sum(w in answer for w in ws) / len(ws)
    in_line = sum(w in lw for w in answer) / (len(answer) or 1)
    return max(in_ans, in_line)


def _pick(lines: list[tuple[str, list[float]]], answer: str) -> list[list[float]]:
    aw = set(_words(answer))
    if not aw:
        return []
    # Một hàng bảng / dòng chữ có thể gồm vài khung (trang scan nghiêng): gom các khung liền nhau cùng chữ.
    groups: list[tuple[str, list[list[float]]]] = []
    for t, box in lines:
        if groups and groups[-1][0] == t:
            groups[-1][1].append(box)
        else:
            groups.append((t, [box]))
    scored = [(_score(t, aw), set(_words(t)) & aw, boxes) for t, boxes in groups]
    best = max((s for s, _, _ in scored), default=0)
    floor = 0.75 if best >= 0.75 else 0.5  # không dòng nào khớp rõ thì nhận dòng khớp nhất nếu đủ gần
    # Chọn theo độ phủ: dòng khớp nhất trước, sau đó chỉ thêm dòng mang thêm ≥ 2 từ / số MỚI của câu trả lời.
    # Nhờ vậy hàng tiêu đề "Condition 1 (Normal)" và các hàng số liệu cùng được tô, còn các hàng na ná nhau
    # (cùng quy cách, khác tên) không thêm gì mới nên không bị tô nhầm.
    covered: set[str] = set()
    hits: list[list[float]] = []
    for s, hit, boxes in sorted(scored, key=lambda x: (-x[0], -len(x[1]))):
        if s < floor or s < best - 0.4:
            break
        new = hit - covered
        if hits and len(new) < 2:
            continue
        covered |= hit
        hits += boxes
    return hits[:80]


def _pdf_lines(p: Path, page: int) -> list[tuple[str, list[float]]] | None:
    """Các dòng chữ của trang PDF kèm khung; None nếu trang không có lớp chữ (trang scan)."""
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(p))
    try:
        pg = pdf[page - 1]
        W, H = pg.get_size()
        tp = pg.get_textpage()
        n = tp.count_chars()
        text = tp.get_text_range() if n else ""
        if len(text.strip()) < rag.MIN_PAGE_TEXT:
            return None
        # chữ của trang theo đúng thứ tự ký tự (mỗi ký tự có khung riêng); cắt thành dòng tại xuống dòng
        rects = []
        for m in re.finditer(r"[^\r\n]+", text if len(text) == n else ""):
            boxes = [tp.get_charbox(i) for i in range(m.start(), m.end())]
            boxes = [bx for bx in boxes if bx[2] > bx[0] and bx[3] > bx[1]]
            if boxes and m.group().strip():
                rects.append((min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes),
                              max(b[3] for b in boxes), m.group().strip()))
    finally:
        pdf.close()
    # gom các đoạn chữ cùng hàng ngang (chồng nhau theo chiều dọc) thành 1 dòng, như 1 hàng của bảng
    rects.sort(key=lambda x: (-x[3], x[0]))
    rows: list[list[Any]] = []
    for l, b, r, t, txt in rects:
        mid = (b + t) / 2
        row = next((rw for rw in rows if rw[1] <= mid <= rw[3]), None)
        if row:
            row[0] = min(row[0], l); row[1] = min(row[1], b); row[2] = max(row[2], r); row[3] = max(row[3], t)
            row[4].append((l, txt))
        else:
            rows.append([l, b, r, t, [(l, txt)]])
    out = []
    for l, b, r, t, parts in rows:
        text = " ".join(x for _, x in sorted(parts))
        out.append((text, [l / W, 1 - t / H, r / W, 1 - b / H]))
    return out


def _ocr_raw(p: Path, page: int) -> str | None:
    db = sqlite3.connect(rag.DB, timeout=10)
    try:
        row = db.execute("SELECT sha1 FROM docs WHERE path=?", (rag.rel(p),)).fetchone()
        sha = row[0] if row else rag.sha1(p)
        # Một trang có thể đã OCR bằng nhiều model (vd. đổi model OCR rồi lập lại chỉ mục). Ưu tiên bản có tọa độ chữ
        # (data-bbox, vd. chandra-ocr) để tô được trên ảnh, dù chỉ mục đang dùng chữ của model khác; không có thì lấy bản mới nhất.
        rows = [r[0] for r in db.execute("SELECT text FROM ocr WHERE sha1=? AND page=? ORDER BY rowid DESC", (sha, page))]
        return next((t for t in rows if "data-bbox" in t), rows[0] if rows else None)
    finally:
        db.close()


def _ink_runs(sub: Any) -> list[tuple[int, int]]:
    """Các dải ngang có chữ trong 1 vùng ảnh (bỏ phần mực nền đều như đường kẻ dọc, khung)."""
    import numpy as np

    frac = (sub < 160).mean(axis=1)
    on = frac > np.percentile(frac, 15) + 0.012
    runs: list[list[int]] = []
    for y, v in enumerate(on):
        if not v:
            continue
        if runs and y - runs[-1][1] <= 2:  # khe 1–2 px trong cùng một dòng chữ
            runs[-1][1] = y + 1
        else:
            runs.append([y, y + 1])
    tall = [r for r in runs if r[1] - r[0] >= 4]
    if not tall:
        return []
    h = float(np.median([r[1] - r[0] for r in tall]))
    return [(a, b) for a, b in tall if b - a >= h * 0.45 and b - a <= h * 2.6]  # bỏ vệt bẩn, đường kẻ ngang dày


def _bands(gray: Any, box: tuple[float, float, float, float], rows: list[str], table: bool) -> list[list[list[float]] | None] | None:
    """Vị trí thật của từng hàng / dòng chữ trong 1 khối OCR, dò trên ảnh trang (model OCR chỉ cho khung cả khối).
    Chia khối thành các dải dọc hẹp, ở mỗi dải đếm các dòng chữ; dải nào có số dòng khớp số hàng có chữ thì dùng
    (hàng tiêu đề nhiều dòng: khớp từ dưới lên). Vị trí mỗi hàng nội suy theo chiều ngang nên bám được trang scan
    hơi nghiêng. Kết quả: mỗi hàng là vài khung nối nhau [x0, y0, x1, y1]; không dò được thì None."""
    import numpy as np

    H, W = gray.shape
    x0, y0, x1, y1 = int(box[0] * W), int(box[1] * H), int(box[2] * W), int(box[3] * H)
    pad = max(2, (y1 - y0) // 50)
    y0, y1 = max(0, y0 - pad), min(H, y1 + pad)
    full = [k for k, r in enumerate(rows) if r.strip()]
    if x1 - x0 < 20 or y1 - y0 < 6 or not full:
        return None
    if not table:
        # Đoạn văn: dòng OCR dài có thể chiếm vài dòng trên giấy. Mỗi dòng OCR nhận ít nhất 1 dòng giấy,
        # số dòng giấy dư chia cho các dòng OCR dài nhất (theo số ký tự).
        runs = _ink_runs(gray[y0:y1, x0:x1])
        if len(runs) < len(full):
            return None
        lens = [max(1, len(rows[k])) for k in full]
        share = [1] * len(full)
        extra = len(runs) - len(full)
        quota = [e * extra / sum(lens) for e in lens]
        for k in range(len(full)):
            share[k] += int(quota[k])
        for k in sorted(range(len(full)), key=lambda k: quota[k] - int(quota[k]), reverse=True)[:len(runs) - sum(share)]:
            share[k] += 1
        out_t: list[list[list[float]] | None] = [None] * len(rows)
        j = 0
        for k, n in zip(full, share):
            a, b = runs[j][0], runs[j + n - 1][1]
            out_t[k] = [[x0 / W, (y0 + a - 2) / H, x1 / W, (y0 + b + 2) / H]]
            j += n
        return out_t
    S = max(1, min(8, (x1 - x0) // 120))
    edges = [x0 + (x1 - x0) * s // S for s in range(S + 1)]
    found = []  # (tâm x của dải, các dòng chữ đã khớp với các hàng có chữ, độ dư)
    for s in range(S):
        runs = _ink_runs(gray[y0:y1, edges[s]:edges[s + 1]])
        d = len(runs) - len(full)
        if 0 <= d <= 3:  # cho dư vài dòng ở trên: hàng tiêu đề nhiều dòng
            found.append(((edges[s] + edges[s + 1]) / 2, runs[d:], d))
    if not found:
        return None
    dmin = min(f[2] for f in found)
    found = [f for f in found if f[2] == dmin]  # chỉ dùng các dải khớp tốt nhất
    xs = np.array([f[0] for f in found])
    out: list[list[list[float]] | None] = [None] * len(rows)
    for j, k in enumerate(full):
        tops = np.array([f[1][j][0] for f in found], float)
        bots = np.array([f[1][j][1] for f in found], float)
        if len(found) >= 2 and np.ptp(xs) > 0:  # đường thẳng qua các dải: theo độ nghiêng của trang
            ct, cb = np.polyfit(xs, tops, 1), np.polyfit(xs, bots, 1)
            top = lambda x, c=ct: c[0] * x + c[1]
            bot = lambda x, c=cb: c[0] * x + c[1]
        else:
            top = lambda x, v=float(np.median(tops)): v
            bot = lambda x, v=float(np.median(bots)): v
        segs = []
        mt, mb = float(np.median(tops)), float(np.median(bots))
        for s in range(S):
            xm = (edges[s] + edges[s + 1]) / 2
            t, b = top(xm), bot(xm)
            if b - t < 3 or abs(t - mt) > (mb - mt) * 1.5:  # đường nội suy lệch bất thường: dùng vị trí trung vị
                t, b = mt, mb
            segs.append([edges[s] / W, (y0 + t - 2) / H, edges[s + 1] / W, (y0 + b + 2) / H])
        out[k] = segs
    return out


def _ocr_lines(raw: str, gray: Any = None) -> list[tuple[str, list[float]]]:
    """Khối chữ OCR có data-bbox (0..1000). Model chỉ cho khung của cả khối; vị trí từng dòng / hàng bảng
    dò trên ảnh trang (đường kẻ, dải chữ), không dò được thì chia đều chiều cao."""
    import lxml.html

    root = lxml.html.fragment_fromstring(raw, create_parent="div")
    for br in root.iter("br"):
        br.tail = "\n" + (br.tail or "")
    out = []
    for block in root:
        bb = block.get("data-bbox")
        if not bb or block.get("data-label") in ("Image", "Figure", "Picture"):
            continue
        try:
            x0, y0, x1, y1 = (float(v) / 1000 for v in bb.split())
        except ValueError:
            continue
        trs = list(block.iter("tr"))
        if trs:
            lines = [" ".join(" ".join(td.text_content().split()) for td in tr if td.tag in ("td", "th")) for tr in trs]
        else:
            lines = [" ".join(x.split()) for x in block.text_content().split("\n")]
            lines = [x for x in lines if x] or [""]
        bands = _bands(gray, (x0, y0, x1, y1), lines, bool(trs)) if gray is not None else None
        h = (y1 - y0) / len(lines)
        for k, text in enumerate(lines):
            segs = bands[k] if bands and bands[k] else [[x0, y0 + k * h, x1, y0 + (k + 1) * h]]
            out.extend((text, sg) for sg in segs)  # 1 hàng có thể là vài khung nối nhau (trang nghiêng)
    return out


def _gray(p: Path, page: int) -> Any:
    import numpy as np
    from PIL import Image

    with Image.open(io.BytesIO(render(p, page))) as im:
        return np.asarray(im.convert("L"))


# ----- dò vị trí chữ trên ảnh trang scan --------------------------------------------------------------
# Model OCR thị giác của Ollama (Qwen-VL…) chỉ trả chữ, không có vị trí. Để tô vàng đúng chỗ trên ảnh, app dò lại vị trí
# từng dòng / ô chữ bằng Windows OCR (có sẵn trong Windows 10/11, chạy gốc trên ARM64, không cần cài model);
# máy không có Windows OCR thì thử RapidOCR (nếu đã cài). Chữ dò được chỉ dùng để định vị: so khớp bỏ dấu (Windows OCR
# thiếu gói ngôn ngữ tiếng Việt thì đọc mất dấu), ưu tiên con số và từ dài. Kết quả lưu SQLite, mỗi trang dò 1 lần.
_engine: Any = None
_engine_lock = threading.Lock()


def _winocr(img: bytes) -> list[dict[str, Any]] | None:
    """Windows OCR (Windows.Media.Ocr): [{t, b (pixel)}] theo từng dòng chữ; None nếu máy không có."""
    try:
        import asyncio

        from winrt.windows.graphics.imaging import BitmapDecoder
        from winrt.windows.media.ocr import OcrEngine
        from winrt.windows.storage.streams import DataWriter, InMemoryRandomAccessStream
    except ImportError:
        return None

    async def run() -> list[dict[str, Any]] | None:
        engine = OcrEngine.try_create_from_user_profile_languages()
        if engine is None:
            return None
        stream = InMemoryRandomAccessStream()
        writer = DataWriter(stream)
        writer.write_bytes(img)
        await writer.store_async()
        writer.detach_stream()
        stream.seek(0)
        bmp = await (await BitmapDecoder.create_async(stream)).get_software_bitmap_async()
        res = await engine.recognize_async(bmp)
        out = []
        for line in res.lines:
            # Windows OCR gom các chữ gần nhau thành 1 dòng; ô bảng cách xa nhau thành các dòng riêng. Tách thêm
            # chỗ hở lớn giữa 2 chữ (> 1,5 lần chiều cao chữ) để mỗi ô bảng là 1 khung.
            words = list(line.words)
            group: list[Any] = []
            for w in words + [None]:
                r = w.bounding_rect if w else None
                if group and (r is None or r.x - (group[-1].bounding_rect.x + group[-1].bounding_rect.width)
                              > 1.5 * max(group[-1].bounding_rect.height, 1)):
                    xs = [g.bounding_rect for g in group]
                    out.append({"t": " ".join(g.text for g in group),
                                "b": [min(q.x for q in xs), min(q.y for q in xs),
                                      max(q.x + q.width for q in xs), max(q.y + q.height for q in xs)]})
                    group = []
                if w:
                    group.append(w)
        return out

    try:
        return asyncio.run(run())
    except OSError:
        return None


def _detect_cells(img: bytes) -> list[dict[str, Any]] | None:
    """Ô chữ trên ảnh (tọa độ 0..1): Windows OCR, không có thì RapidOCR; None nếu không dò được."""
    from PIL import Image

    with Image.open(io.BytesIO(img)) as im:
        W, H = im.size
    cells = _winocr(img)
    if cells is None:
        global _engine
        try:
            if _engine is None:
                from rapidocr_onnxruntime import RapidOCR

                _engine = RapidOCR()
        except ImportError:
            return None
        import numpy as np

        with Image.open(io.BytesIO(img)) as im:
            res, _ = _engine(np.asarray(im.convert("RGB")))
        cells = [{"t": text, "b": [min(p[0] for p in box), min(p[1] for p in box), max(p[0] for p in box), max(p[1] for p in box)]}
                 for box, text, _c in res or []]
    return [{"t": c["t"], "b": [round(c["b"][0] / W, 4), round(c["b"][1] / H, 4), round(c["b"][2] / W, 4), round(c["b"][3] / H, 4)]}
            for c in cells]
NUM = re.compile(r"\d+(?:[.,]\d+)*")
TOK = re.compile(r"\d+(?:[.,]\d+)*|[^\W\d_]+", re.U)


def _fold(s: str) -> str:
    s = unicodedata.normalize("NFD", s.replace("đ", "d").replace("Đ", "D"))
    return "".join(ch for ch in s if unicodedata.category(ch) != "Mn").lower()


def _detect(p: Path, page: int) -> list[dict[str, Any]] | None:
    """Các ô chữ trên ảnh trang: [{t: chữ, b: [x0, y0, x1, y1] (0..1)}]; None nếu máy không dò được (không có OCR)."""
    db = sqlite3.connect(rag.DB, timeout=10)
    try:
        db.execute("CREATE TABLE IF NOT EXISTS layout (sha1 TEXT, page INTEGER, data TEXT, PRIMARY KEY (sha1, page))")
        row = db.execute("SELECT sha1 FROM docs WHERE path=?", (rag.rel(p),)).fetchone()
        sha = row[0] if row else rag.sha1(p)
        hit = db.execute("SELECT data FROM layout WHERE sha1=? AND page=?", (sha, page)).fetchone()
        if hit:
            return json.loads(hit[0])
        with _engine_lock:
            cells = _detect_cells(render(p, page))
        if cells is None:
            return None
        with db:
            db.execute("INSERT OR REPLACE INTO layout VALUES (?, ?, ?)", (sha, page, json.dumps(cells, ensure_ascii=False)))
        return cells
    finally:
        db.close()


def _rows(cells: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Gom ô chữ thành hàng: duyệt từ trái sang phải, nối vào hàng có ô bên phải cùng cao độ (bám độ nghiêng trang scan)."""
    rows: list[list[dict[str, Any]]] = []
    for c in sorted(cells, key=lambda c: c["b"][0]):
        x0, y0, x1, y1 = c["b"]
        mid, h = (y0 + y1) / 2, y1 - y0
        best, gap = None, h * 0.6
        for r in rows:
            last = r[-1]["b"]
            d = abs((last[1] + last[3]) / 2 - mid)
            if d < gap and last[2] <= x0 + h:
                best, gap = r, d
        if best is None:
            rows.append([c])
        else:
            best.append(c)
    return rows


def _match(text: str, toks: set[str]) -> tuple[set[str], float]:
    """Các từ / số của câu trả lời có trong 1 hàng chữ dò được, và tỉ lệ chữ của hàng được câu trả lời phủ."""
    s = _fold(text)  # các ô nối bằng khoảng trắng; trong 1 ô, chữ dò được có thể dính liền nhau
    found: set[str] = set()
    covered = [False] * len(s)
    for t in toks:
        if NUM.fullmatch(t):  # số: không được dính liền chữ số khác (26.5 ≠ 126.5)
            pat = r"(?<![\d.,])" + re.escape(t) + r"(?![\d]|[.,]\d)"
        elif len(t) >= 4:  # từ dài: tìm như chuỗi con ("pressure" trong "opressure")
            pat = re.escape(t)
        else:  # từ ngắn: phải đứng riêng, tránh "la" khớp bừa trong "plant"
            pat = r"\b" + re.escape(t) + r"\b"
        ms = list(re.finditer(pat, s))
        if ms:
            found.add(t)
            for m in ms:
                for k in range(m.start(), m.end()):
                    covered[k] = True
    alnum = [k for k, ch in enumerate(s) if ch.isalnum()]
    return found, (sum(covered[k] for k in alnum) / len(alnum)) if alnum else 0.0


def _pick_detected(cells: list[dict[str, Any]], answer: str) -> list[list[float]]:
    """Chọn hàng chữ khớp câu trả lời và trả khung từng ô chữ của các hàng đó (tô đúng ô, cả khi trang nghiêng)."""
    toks = {t for t in TOK.findall(_fold(answer)) if NUM.fullmatch(t) and len(re.sub(r"\D", "", t)) >= 2 or not NUM.fullmatch(t) and len(t) >= 2}
    scored = []
    for r in _rows(cells):
        found, cov = _match(" ".join(c["t"] for c in r), toks)
        nums = sum(1 for t in found if NUM.fullmatch(t))
        # hàng bảng có ≥ 2 con số của câu trả lời, hoặc dòng chữ phần lớn nằm trong câu trả lời
        if nums >= 2 or cov >= 0.75 and len(found) >= 2 or nums >= 1 and cov >= 0.5:
            scored.append((len(found) + nums, cov, found, r))
    covered: set[str] = set()
    out: list[list[float]] = []
    for _n, _cov, found, r in sorted(scored, key=lambda x: (-x[0], -x[1])):
        if out and len(found - covered) < 2:  # hàng na ná hàng đã chọn, không thêm thông tin mới
            continue
        covered |= found
        out += [c["b"] for c in r]
    return out[:120]


def marks(p: Path, page: int, answer: str) -> dict[str, Any]:
    src, lines = "pdf", None
    if kind(p) == "pdf":
        lines = _pdf_lines(p, page)
    if lines is not None:  # PDF có chữ: vị trí chính xác từ lớp chữ
        boxes = _pick(lines, answer) if answer.strip() else []
        return {"page": page, "pages": page_count(p), "source": src, "boxes": [[round(v, 4) for v in b] for b in boxes]}
    boxes: list[list[float]] = []
    src = "detect"
    if answer.strip():
        cells = _detect(p, page)
        if cells:
            boxes = _pick_detected(cells, answer)
        if not boxes:  # không dò được / không khớp: dùng khung khối chữ của model OCR (vd. chandra-ocr) nếu có
            raw = _ocr_raw(p, page)
            src = "ocr"
            lines = _ocr_lines(raw, _gray(p, page)) if raw and re.search(r"data-bbox", raw) else []
            if not lines and cells is None:
                src = "ocr-nobox"  # không có RapidOCR lẫn tọa độ từ model OCR: không tô được trên ảnh
            boxes = _pick(lines, answer)
    return {"page": page, "pages": page_count(p), "source": src, "boxes": [[round(v, 4) for v in b] for b in boxes]}

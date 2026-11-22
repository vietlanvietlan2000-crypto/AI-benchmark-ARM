"""API của chatbot hỏi đáp tài liệu. Phần tìm tài liệu nằm ở rag.py."""
from __future__ import annotations

import asyncio
import io
import json
import os
import re
import time
from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response, StreamingResponse

from . import pages, rag, websearch
from .core import LM_URL
from .gpu import gpu_info
from .ollama import KEEP_ALIVE, Ollama, can_think, ctx_of, set_ctx

router = APIRouter(prefix="/api/chat")
busy: Callable[[], Any] = lambda: None  # server.py gán: trả job benchmark đang chạy (nếu có)
indexing: dict[str, Any] = {"running": False, "stage": "", "done": 0, "total": 0, "file": "", "error": None, "cancel": False}
_index: rag.Index | None = None

CTX_WANT = 16384  # ngữ cảnh khi chatbot tự nạp model (đủ để đọc trọn tài liệu vài trang)
EMBED_CTX = 4096  # ngữ cảnh của model embedding: mỗi đoạn tài liệu chỉ dài khoảng 1.200 ký tự
OCR_CTX = 16384  # ngữ cảnh của model OCR: ảnh 1 trang ~2.200 token + chữ đọc ra, nhân số trang OCR cùng lúc
SYSTEM = (
    "Bạn là trợ lý tra cứu thân thiện, trả lời như một đồng nghiệp am hiểu tài liệu: tự nhiên, rõ ràng, đi thẳng vào ý chính. "
    "Thông tin thực tế (quy định, số liệu, thông số, quy trình…) chỉ lấy từ các NGUỒN THAM KHẢO được đánh số trong tin nhắn; "
    "sau mỗi ý lấy từ nguồn, ghi số nguồn trong ngoặc vuông, ví dụ [2] hoặc [1][3]. "
    "Nếu các nguồn không có thông tin cần hỏi, nói thẳng là chưa tìm thấy trong tài liệu và gợi ý cách hỏi khác, không tự bịa. "
    "Câu chào hỏi, cảm ơn hay trao đổi xã giao thì đáp lại tự nhiên, không cần nhắc tới nguồn. "
    "Trả lời bằng ngôn ngữ của người dùng."
)
SYSTEM_PLAIN = "Bạn là trợ lý AI thân thiện, hữu ích. Trò chuyện tự nhiên, trả lời rõ ràng, bằng ngôn ngữ của người dùng."
SYSTEM_CHAT = (
    "Bạn là trợ lý AI thân thiện của doanh nghiệp, có thể tra cứu kho tài liệu nội bộ{docs}. "
    "Tin nhắn này không cần tra tài liệu: hãy trò chuyện tự nhiên, ngắn gọn, ấm áp, bằng ngôn ngữ của người dùng. "
    "Khi được hỏi bạn làm được gì: giới thiệu bạn tra cứu tài liệu nội bộ và trích dẫn trang/mục, lập bảng, xuất Excel, vẽ biểu đồ"
    "{web}. Đừng tự bịa nội dung tài liệu; nếu người dùng cần thông tin trong tài liệu, mời họ hỏi cụ thể."
)
ROUTER = (
    "Bạn là bộ phân loại tin nhắn cho một trợ lý tra cứu tài liệu nội bộ{docs}. Nguồn tra cứu đang bật: {sources}.\n"
    "Đọc đoạn hội thoại và TIN NHẮN CUỐI của người dùng, trả về DUY NHẤT một dòng JSON:\n"
    '{{"action": "chat" hoặc "search", "query": "...", "query_en": "..."}}\n'
    '- "chat": chào hỏi, cảm ơn, tạm biệt, khen chê, hỏi bạn là ai / làm được gì, tán gẫu; yêu cầu viết lại, dịch, tóm tắt, '
    "rút gọn, đổi định dạng, lập bảng, vẽ biểu đồ từ nội dung ĐÃ CÓ trong hội thoại.\n"
    '- "search": cần thông tin thực tế (quy định, chính sách, số liệu, thông số kỹ thuật, quy trình, sản phẩm, giá…) '
    'hoặc hỏi về nội dung tài liệu. Khi phân vân, chọn "search".\n'
    '- "query" (khi search): viết lại tin nhắn cuối thành câu tìm kiếm đầy đủ, tự hiểu được mà không cần hội thoại '
    "(thay \"nó\", \"cái đó\", \"còn … thì sao\" bằng đối tượng cụ thể), giữ nguyên ngôn ngữ và thuật ngữ, mã hiệu.\n"
    '- "query_en" (khi search): cùng câu tìm kiếm đó dịch sang tiếng Anh theo cách viết của tài liệu kỹ thuật '
    '(vd. "áp suất" → "pressure", "chế độ bình thường" → "normal condition", "nhiệt độ đầu vào" → "inlet temperature"), '
    "giữ nguyên mã thiết bị, ký hiệu, con số. Nếu tin nhắn đã là tiếng Anh thì dịch sang tiếng Việt.\n"
    "Không giải thích, không thêm chữ nào ngoài JSON."
)
# Câu xã giao rõ ràng: không cần hỏi model phân loại
SMALLTALK = re.compile(
    r"^\s*(hi+|hello+|hey+|helo|alo+|chào|xin chào|chào (bạn|em|anh|chị|bot|ad)|hế lô|hi there|good (morning|afternoon|evening)|"
    r"cảm ơn|cám ơn|cảm ơn (bạn|em|nhiều|nha|nhé)|thanks?( you)?|thank you( so much)?|tks|ok+|oke+|okay|được rồi|tốt lắm|tuyệt|"
    r"bye|tạm biệt|bái bai|hẹn gặp lại|bạn (là ai|tên gì)|you are who|who are you)"
    r"[\s!.?,~:)(^❤️👍😊🙂]*((bạn|nhé|nha|nhiều|ạ|à|ơi|bot|em|anh|chị|so much|a lot)[\s!.?,~:)(^❤️👍😊🙂]*){0,3}$", re.I)
FORMAT = (
    " Khi người dùng muốn bảng số liệu hoặc file Excel: trình bày số liệu bằng bảng Markdown (giao diện có sẵn nút xuất Excel "
    "cho mỗi bảng, đừng nói rằng bạn không tạo được file). Khi người dùng muốn biểu đồ: viết một khối mã ```chart chứa JSON dạng "
    '{"type": "bar", "title": "...", "labels": ["A", "B"], "series": [{"name": "...", "data": [1, 2]}]} '
    '("type" là "bar", "line" hoặc "pie"; "data" là số thuần, không có dấu phân cách hàng nghìn hay đơn vị).'
)


def index() -> rag.Index:
    global _index
    if _index is None:
        _index = rag.Index.load()
    return _index


def _check_idle() -> None:
    if busy():
        raise HTTPException(409, "Đang có một bài test chạy, hãy đợi hoặc huỷ nó trước")
    if indexing["running"]:
        raise HTTPException(409, "Đang lập chỉ mục tài liệu, hãy đợi xong hoặc bấm Dừng")


@router.get("/state")
async def state() -> dict[str, Any]:
    def work() -> dict[str, Any]:
        ix = index()
        stale = set(ix.stale())
        files = []
        for p in rag.list_files():
            r = rag.rel(p)
            f = ix.files.get(r) or {}
            files.append({"path": r, "kb": round(p.stat().st_size / 1024, 1), "indexed": r in ix.files and r not in stale,
                          "chunks": f.get("n", 0), "note": f.get("note", ""), "ocr_pages": f.get("ocr_pages", 0),
                          "ocr_missing": f.get("ocr_missing", 0)})
        return {"docs_dir": str(rag.DOCS), "exts": sorted(rag.EXTS), "files": files, "embed_model": ix.model,
                "chunks": len(ix.chunks), "indexing": indexing}

    return await asyncio.to_thread(work)


@router.post("/index")
async def build_index(body: dict[str, Any]) -> dict[str, Any]:
    if indexing["running"]:
        return {"ok": True}
    _check_idle()
    model, ocr_model = str(body.get("embed_model") or ""), str(body.get("ocr_model") or "")
    if not model:
        raise HTTPException(400, "Chưa chọn model embedding")
    indexing.update(running=True, stage="scan", done=0, total=0, file="", error=None, cancel=False)

    async def prepare_ocr(m: str) -> None:
        """Gỡ mọi model khác để model OCR có đủ bộ nhớ, rồi nạp model OCR."""
        lm = Ollama(LM_URL)
        try:
            if m not in {x["id"] for x in await lm.list_models()}:
                raise RuntimeError(f"Ollama không có model {m}. Tải bằng: ollama pull {m}")
            await lm.unload_all(keep={m})
            await lm.load(m, OCR_CTX)
        finally:
            await lm.close()

    async def prepare_embed(m: str) -> None:
        if ocr_model and ocr_model != m:  # OCR xong thì gỡ model OCR để nhường bộ nhớ
            lm = Ollama(LM_URL)
            try:
                await lm.unload(ocr_model)
            finally:
                await lm.close()
        await _prepare_embed(m)

    async def work() -> None:
        global _index
        try:
            _index = await rag.build(model, ocr_model, indexing, prepare_ocr, prepare_embed)
        except rag.Cancelled:
            indexing["error"] = "Đã dừng. Phần đã làm (trang đã OCR, tài liệu đã lưu) được giữ lại, bấm Cập nhật chỉ mục để làm tiếp."
        except Exception as e:  # noqa: BLE001
            indexing["error"] = str(e) or type(e).__name__
        finally:
            if indexing["error"]:
                _index = None  # đọc lại từ cơ sở dữ liệu những tài liệu đã lưu được
            indexing.update(running=False, stage="", cancel=False)

    asyncio.get_running_loop().create_task(work())
    return {"ok": True}


@router.post("/index/cancel")
async def cancel_index() -> dict[str, Any]:
    if indexing["running"]:
        indexing["cancel"] = True
    return {"ok": True}


@router.post("/open")
async def open_doc(body: dict[str, Any]) -> dict[str, Any]:
    """Mở thư mục tài liệu (path rỗng) hoặc 1 tài liệu trong đó."""
    rag.DOCS.mkdir(exist_ok=True)
    path = (rag.DOCS / str(body.get("path", ""))).resolve()
    if not path.exists() or not (path == rag.DOCS or path.is_relative_to(rag.DOCS)):
        raise HTTPException(400, "Không mở được file này")
    os.startfile(path)  # noqa: S606
    return {"ok": True}


async def _prepare_embed(model: str) -> None:
    """Nạp model embedding với ngữ cảnh nhỏ (mỗi đoạn tài liệu chỉ ~1.200 ký tự): ngữ cảnh lớn tốn bộ nhớ vô ích
    và không còn chỗ cho model trả lời."""
    lm = Ollama(LM_URL)
    try:
        info = next((m for m in await lm.list_models() if m["id"] == model), None)
        if info is None:
            raise RuntimeError(f"Ollama không có model {model}. Tải bằng: ollama pull {model}")
        set_ctx(model, min(EMBED_CTX, info.get("max_context_length") or EMBED_CTX))
        if info.get("state") != "loaded":
            try:
                await lm.load(model)
            except RuntimeError as e:
                raise RuntimeError(f"Không nạp được model embedding {model}. Có thể bộ nhớ đã đầy: hãy bấm \"Gỡ các model đã nạp\" "
                                   f"hoặc chọn model embedding nhỏ hơn. ({e})") from e
    finally:
        await lm.close()


CTX_STEPS = (65536, 32768, 16384, 8192)  # các mức ngữ cảnh thử, lớn trước
ANSWER_MAX_TOKENS = 8192  # trần token (suy nghĩ + trả lời) khi trả lời từ tài liệu; chạm trần mà chưa trả lời thì hỏi lại không suy nghĩ
KV_MB_PER_GB_PER_K = 6.0  # bộ nhớ ngữ cảnh ~6 MB cho mỗi 1.024 token trên mỗi GB model (đo trên Qwen3.8 27B: ~124 MB / 1K token)
_ctx_checked: set[tuple[str, int]] = set()  # (model, ngữ cảnh đang dùng) đã xét tăng ngữ cảnh, khỏi tính lại mỗi câu hỏi


def _need_gb(info: dict[str, Any], ctx: int) -> float:
    """Bộ nhớ (GB) ước lượng để chạy model với ngữ cảnh ctx: file model +5% +0,3 GB + bộ nhớ ngữ cảnh.
    Ollama không có lệnh ước lượng như LM Studio nên dùng công thức (lệch về phía an toàn)."""
    size = (info.get("size_bytes") or 0) / 2**30
    return size * 1.05 + 0.3 + size * KV_MB_PER_GB_PER_K * ctx / 1024 / 1024


async def _best_ctx(info: dict[str, Any], held_gb: float = 0.0) -> int:
    """Ngữ cảnh lớn nhất (trong CTX_STEPS) mà bộ nhớ còn trống chứa được. held_gb: bộ nhớ model này đang chiếm
    (được trả lại khi nạp lại). Máy bộ nhớ chung (RTX Spark) chừa thêm 4 GB cho Windows và ứng dụng."""
    g = await asyncio.to_thread(gpu_info)
    total, used = g.get("vram_total_mb"), g.get("vram_used_mb")
    max_ctx = int(info.get("max_context_length") or CTX_WANT)
    if not total or used is None or not info.get("size_bytes"):
        return min(CTX_WANT, max_ctx)
    free = (total - used) / 1024 + held_gb - (5.5 if g.get("unified") else 1.5)
    for c in CTX_STEPS:
        if c <= max_ctx and _need_gb(info, c) <= free:
            return c
    return min(CTX_STEPS[-1], max_ctx)


async def _prepare_model(model: str) -> int:
    """Bảo đảm model chat đã nạp với ngữ cảnh đủ lớn theo bộ nhớ; trả về độ dài ngữ cảnh (token) đang dùng."""
    lm = Ollama(LM_URL)
    try:
        models = await lm.list_models()
        info = next((m for m in models if m["id"] == model), None)
        if info is None:
            raise RuntimeError(f"Ollama không có model {model}. Tải bằng: ollama pull {model}")
        if info.get("state") != "loaded":
            for m in models:  # gỡ model chat khác để nhường bộ nhớ, giữ lại model embedding
                if m.get("state") == "loaded" and m.get("type") != "embeddings":
                    await lm.unload(m["id"])
            await lm.load(model, await _best_ctx(info))
        else:
            # Đã nạp sẵn với ngữ cảnh nhỏ (vd. do benchmark nạp): nếu còn dư bộ nhớ thì nạp lại với ngữ cảnh lớn hơn.
            cur = int(info.get("loaded_context_length") or ctx_of(model))
            if cur < CTX_STEPS[0] and (model, cur) not in _ctx_checked:
                _ctx_checked.add((model, cur))
                want = await _best_ctx(info, _need_gb(info, cur))
                if want > cur:
                    await lm.unload(model)
                    await lm.load(model, want)
                else:
                    set_ctx(model, cur)
            else:
                set_ctx(model, cur)
        return ctx_of(model)
    finally:
        await lm.close()


@router.get("/doc")
async def get_doc(path: str) -> dict[str, Any]:
    """Nội dung 1 tài liệu theo từng đoạn đã lập chỉ mục, để giao diện hiện và tô sáng đoạn được trích dẫn."""
    chunks = index().of(path)
    if not chunks:
        raise HTTPException(404, "Tài liệu này chưa có trong chỉ mục")
    return {"path": path, "chunks": [{"i": c["i"], "loc": c.get("loc"), "text": c["text"], "ocr": c.get("ocr", False)} for c in chunks]}


@router.get("/page")
async def get_page(path: str, page: int = 1) -> Response:
    """Ảnh 1 trang của tài liệu gốc (PDF / file ảnh)."""
    try:
        p = pages.resolve(path)
        if not pages.kind(p) or not 1 <= page <= pages.page_count(p):
            raise FileNotFoundError(path)
        jpg = await asyncio.to_thread(pages.render, p, page)
    except FileNotFoundError:
        raise HTTPException(404, "Không có trang này") from None
    return Response(jpg, media_type="image/jpeg", headers={"Cache-Control": "max-age=300"})


@router.post("/marks")
async def get_marks(body: dict[str, Any]) -> dict[str, Any]:
    """Khung tô vàng trên 1 trang: các dòng khớp với câu trả lời. body: {path, page, answer}."""
    try:
        p = pages.resolve(str(body.get("path", "")))
        if not pages.kind(p):
            raise FileNotFoundError
        return await asyncio.to_thread(pages.marks, p, int(body.get("page") or 1), str(body.get("answer") or ""))
    except FileNotFoundError:
        raise HTTPException(404, "Tài liệu này không xem được dạng trang") from None


def _num(v: Any) -> Any:
    """Chữ trông như số (kiểu Việt: 6.490.000 hoặc 5,5) thì đổi thành số để Excel tính toán được."""
    if not isinstance(v, str):
        return v
    t = v.strip().replace("\u00a0", "")
    if re.fullmatch(r"-?\d{1,3}(\.\d{3})+", t):
        return int(t.replace(".", ""))
    if re.fullmatch(r"-?\d+", t) and (len(t) < 2 or t.lstrip("-")[0] != "0"):
        return int(t)
    if re.fullmatch(r"-?\d+,\d+", t):
        return float(t.replace(",", "."))
    return v


@router.post("/excel")
async def excel(body: dict[str, Any]) -> Response:
    """Tạo file Excel từ 1 bảng (rows) hoặc 1 biểu đồ (chart: kèm biểu đồ Excel thật)."""
    from openpyxl import Workbook
    from openpyxl.chart import BarChart, LineChart, PieChart, Reference
    from openpyxl.styles import Font
    from openpyxl.utils import get_column_letter

    chart = body.get("chart")
    if chart:
        series = chart.get("series") or []
        rows = [["", *[str(x.get("name") or f"Dãy {k + 1}") for k, x in enumerate(series)]]]
        for i, lab in enumerate(chart.get("labels") or []):
            rows.append([lab, *[(x.get("data") or [None] * (i + 1))[i] if i < len(x.get("data") or []) else None for x in series]])
    else:
        rows = body.get("rows") or []
    if len(rows) < 2:
        raise HTTPException(400, "Không có dữ liệu để xuất")
    wb = Workbook()
    ws = wb.active
    ws.title = "Du lieu"
    for r in rows:
        ws.append([_num(c) for c in r])
    for c in ws[1]:
        c.font = Font(bold=True)
    for k in range(1, ws.max_column + 1):
        width = max(len(str(r[k - 1])) if k <= len(r) and r[k - 1] is not None else 0 for r in rows)
        ws.column_dimensions[get_column_letter(k)].width = min(60, max(10, width + 2))
    if chart:
        kind = chart.get("type")
        ch = PieChart() if kind == "pie" else LineChart() if kind == "line" else BarChart()
        ch.title = chart.get("title") or None
        ch.add_data(Reference(ws, min_col=2, max_col=ws.max_column, min_row=1, max_row=ws.max_row), titles_from_data=True)
        ch.set_categories(Reference(ws, min_col=1, min_row=2, max_row=ws.max_row))
        ch.width, ch.height = 20, 11
        ws.add_chart(ch, f"{get_column_letter(ws.max_column + 2)}2")
    buf = io.BytesIO()
    wb.save(buf)
    return Response(buf.getvalue(), media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


async def _web_passages(question: str, budget: int) -> list[dict[str, Any]]:
    """Tìm trên internet và chọn những đoạn trùng nhiều từ với câu hỏi nhất ở mỗi trang."""
    qw = rag.words(question)
    out: list[dict[str, Any]] = []
    results = await websearch.gather(question)
    per_page = max(600, budget // max(1, len(results)))
    for r in results:
        parts = rag.split(r["text"]) if r["text"] else []
        parts.sort(key=lambda t: -len(qw & rag.words(t)))  # sort ổn định: bằng điểm thì đoạn đầu trang đứng trước
        picked, used = [], 0
        for t in parts[:3]:
            if used + len(t) > per_page:
                break
            picked.append(t)
            used += len(t)
        if not picked and r["snippet"]:
            picked = [r["snippet"]]
        out += [{"kind": "web", "url": r["url"], "title": r["title"], "text": t} for t in picked]
    return out


def _catalog(limit: int = 30) -> str:
    """Tên ngắn các tài liệu đang có, để model biết kho tài liệu nói về gì."""
    names = sorted(index().files)
    if not names:
        return ""
    short = [re.sub(r"\.\w+$", "", n.rsplit("/", 1)[-1]) for n in names[:limit]]  # chỉ tên file, không đường dẫn
    more = f" và {len(names) - limit} tài liệu khác" if len(names) > limit else ""
    return " (gồm: " + "; ".join(short) + more + ")"


async def _route(model: str, history: list[dict[str, Any]], question: str, use_docs: bool, use_web: bool) -> tuple[str, str, str]:
    """Phân loại tin nhắn: ("chat", "", "") nếu không cần tra cứu; ("search", câu tìm kiếm đã viết lại, câu đó bằng
    tiếng Anh) nếu cần — tài liệu kỹ thuật thường viết tiếng Anh nên tìm bằng cả hai ngôn ngữ.
    Câu xã giao rõ ràng nhận ra ngay; còn lại hỏi nhanh model đang dùng. Lỗi hay trả lời lạ thì coi như cần tra cứu."""
    if SMALLTALK.match(question):
        return "chat", "", ""
    sources = " và ".join(x for x, on in (("tài liệu nội bộ", use_docs), ("internet", use_web)) if on)
    convo = "\n".join(f"{'Người dùng' if m['role'] == 'user' else 'Trợ lý'}: {m['content'][:600]}" for m in history[-4:])
    payload = {"model": model, "stream": False, "format": "json", "keep_alive": KEEP_ALIVE,
               "options": {"temperature": 0, "num_predict": 600, "num_ctx": ctx_of(model)},
               "messages": [{"role": "system", "content": ROUTER.format(docs=_catalog() if use_docs else "", sources=sources)},
                            {"role": "user", "content": (f"HỘI THOẠI TRƯỚC:\n{convo}\n\n" if convo else "") + f"TIN NHẮN CUỐI: {question}"}]}
    if await can_think(model):
        payload["think"] = False
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(120, connect=5)) as http:
            r = await http.post(f"{LM_URL}/api/chat", json=payload)
        text = (r.json().get("message") or {}).get("content") or ""
        m = re.search(r"\{.*\}", text, re.S)
        obj = json.loads(m.group()) if m else {}
    except Exception:  # noqa: BLE001
        return "search", "", ""
    if str(obj.get("action", "")).lower() == "chat":
        return "chat", "", ""
    return "search", str(obj.get("query") or "").strip(), str(obj.get("query_en") or "").strip()


@router.post("/ask")
async def ask(body: dict[str, Any]) -> StreamingResponse:
    _check_idle()
    model = str(body.get("model") or "")
    msgs = [m for m in body.get("messages") or [] if m.get("role") in ("user", "assistant") and m.get("content")]
    if not model or not msgs or msgs[-1]["role"] != "user":
        raise HTTPException(400, "Thiếu model hoặc câu hỏi")
    use_docs, use_web = bool(body.get("use_docs", True)), bool(body.get("use_web"))

    async def gen() -> AsyncIterator[str]:
        def ev(**k: Any) -> str:
            return json.dumps(k, ensure_ascii=False) + "\n"

        try:
            yield ev(type="status", text="Đang nạp model…")
            ctx = await _prepare_model(model)
            history, question = msgs[-7:-1], msgs[-1]["content"]
            prev = next((m["content"] for m in reversed(history) if m["role"] == "user"), "")
            used = sum(len(m["content"]) for m in history) + len(question)
            # Chừa khoảng 1/3 ngữ cảnh cho phần model suy nghĩ + trả lời (model suy nghĩ như Qwen, gpt-oss dùng cả nghìn token
            # trước khi viết chữ đầu tiên); ~2 ký tự / token (tiếng Việt có dấu và bảng số liệu tốn token hơn văn tiếng Anh).
            reserve = min(8192, max(3000, ctx // 3))
            budget = max(1500, min(40000, int((ctx - reserve - 1000) * 2.0) - used))
            passages: list[dict[str, Any]] = []
            action, query, query_en = "plain", "", ""  # plain: người dùng tắt cả tài liệu lẫn internet
            if use_docs or use_web:
                yield ev(type="status", text="Đang hiểu câu hỏi…")
                action, query, query_en = await _route(model, history, question, use_docs, use_web)
                yield ev(type="route", mode=action, query=query)
            chatting = action == "chat"
            docs_on, web_on = use_docs and not chatting, use_web and not chatting
            if docs_on:
                ix = index()
                if not ix.chunks:
                    raise RuntimeError("Chưa có chỉ mục tài liệu. Hãy bấm \"Cập nhật chỉ mục\" trước.")
                yield ev(type="status", text="Đang tìm tài liệu liên quan…")
                await _prepare_embed(ix.model)
                # câu tìm kiếm đã được viết lại cho đủ ý; không có thì ghép câu trước để hiểu câu hỏi nối tiếp
                doc_queries = [f"{query}\n{question}" if query else f"{prev}\n{question}".strip(), query_en]
                found = await rag.retrieve(ix, doc_queries, budget * 6 // 10 if web_on else budget)
                passages += [{"kind": "doc", "doc": c["doc"], "i": c["i"], "loc": c.get("loc"), "text": c["text"]} for c in found]
            if web_on:
                yield ev(type="status", text="Đang tìm kiếm trên internet…")
                q = query or (question if len(question) > 25 or not prev else f"{prev} {question}")
                passages += await _web_passages(q, budget - sum(len(x["text"]) for x in passages))
            def build(passages: list[dict[str, Any]]) -> tuple[str, str]:
                """Tin nhắn gửi model: nguồn đánh số, gom theo tài liệu (tên tài liệu ghi 1 lần, mỗi đoạn chỉ ghi trang/mục)."""
                if chatting:  # trò chuyện: không đưa nguồn, để model đáp tự nhiên
                    return question, SYSTEM_CHAT.format(docs=_catalog(12) if use_docs else "",
                                                        web=", tìm kiếm trên internet" if use_web else "")
                if not (docs_on or web_on):
                    return question, SYSTEM_PLAIN
                blocks, last = [], None
                for x in passages:
                    if x["kind"] == "doc":
                        if x["doc"] != last:
                            blocks.append(f"=== Tài liệu: {x['doc']} ===")
                            last = x["doc"]
                        where = rag.loc_text(x["loc"])
                        blocks.append(f"[{x['n']}]{' (' + where + ')' if where else ''}\n{x['text']}")
                    else:
                        last = None
                        blocks.append(f"[{x['n']}] Trang web: {x['title']} ({x['url']})\n{x['text']}")
                content = "NGUỒN THAM KHẢO:\n" + ("\n\n".join(blocks) or "(không tìm thấy nguồn nào)") + f"\n\nCÂU HỎI: {question}"
                return content, SYSTEM + (f" Hôm nay là ngày {time.strftime('%d/%m/%Y')}." if use_web else "")

            t0, tokens, think_off = time.perf_counter(), 0, False
            thinker = await can_think(model)  # chỉ gửi "think" cho model có chế độ suy nghĩ (model khác Ollama báo lỗi)
            for attempt in range(2):
                for n, x in enumerate(passages, 1):
                    x["n"] = n
                yield ev(type="sources", ctx=ctx, sources=[{k: v for k, v in x.items() if k != "text" or x["kind"] == "web"} for x in passages])
                content, system = build(passages)
                options = {"temperature": 0.6 if chatting else 0.2, "num_ctx": ctx, "num_predict": -1}
                payload: dict[str, Any] = {"model": model, "stream": True, "keep_alive": KEEP_ALIVE, "options": options,
                                           "messages": [{"role": "system", "content": system + FORMAT}, *history,
                                                        {"role": "user", "content": content}]}
                if (chatting or think_off) and thinker:  # trò chuyện / lần hỏi lại: không cho model suy nghĩ lâu, trả lời ngay
                    payload["think"] = False
                elif docs_on or web_on:  # giới hạn suy nghĩ + trả lời (~1 phút): model không bị kẹt suy nghĩ cả chục nghìn token
                    options["num_predict"] = ANSWER_MAX_TOKENS
                thinking, wrote, finish, used_ctx = False, False, None, 0
                async with httpx.AsyncClient(timeout=httpx.Timeout(900, connect=5)) as http, \
                        http.stream("POST", f"{LM_URL}/api/chat", json=payload) as r:
                    if r.status_code != 200:
                        raise RuntimeError(f"Ollama trả lỗi {r.status_code}: {(await r.aread()).decode('utf-8', 'replace')[:300]}")
                    async for line in r.aiter_lines():
                        if not line.strip():
                            continue
                        obj = json.loads(line)
                        if obj.get("error"):
                            raise RuntimeError(f"Ollama báo lỗi: {obj['error']}")
                        msg = obj.get("message") or {}
                        if msg.get("content"):
                            wrote = True
                            yield ev(type="text", text=msg["content"])
                        elif msg.get("thinking") and not thinking:
                            thinking = True
                            yield ev(type="status", text="Model đang suy nghĩ…")
                        if obj.get("done"):
                            finish = obj.get("done_reason") or "stop"
                            tokens = obj.get("eval_count") or 0
                            used_ctx = (obj.get("prompt_eval_count") or 0) + tokens
                if wrote:
                    break
                # Model không viết chữ nào: suy nghĩ chạm giới hạn token, hết ngữ cảnh, hoặc suy nghĩ xong mà không trả lời
                # (câu hỏi rộng làm model suy nghĩ rất dài). Hỏi lại 1 lần với chế độ suy nghĩ TẮT; nếu là hết ngữ cảnh
                # thì giảm nửa số đoạn tài liệu.
                overflow = finish == "length" and used_ctx >= ctx - 256
                if attempt == 0:
                    yield ev(type="status", text="Tài liệu dài quá ngữ cảnh, đang thử lại với ít đoạn hơn…" if overflow
                             else "Model suy nghĩ quá lâu, đang hỏi lại để trả lời trực tiếp…")
                    if overflow and docs_on and passages:
                        half = sum(len(x["text"]) for x in passages) // 2
                        web = [x for x in passages if x["kind"] == "web"]
                        found = await rag.retrieve(ix, doc_queries, max(1500, half - sum(len(x["text"]) for x in web)))
                        passages = [{"kind": "doc", "doc": c["doc"], "i": c["i"], "loc": c.get("loc"), "text": c["text"]} for c in found] + web
                    think_off = True
                    continue
                if not overflow:
                    raise RuntimeError("Model trả lời rỗng")
                raise RuntimeError(f"Model dùng hết ngữ cảnh ({ctx} token) khi đang suy nghĩ nên chưa kịp trả lời. "
                                   "Hãy gỡ bớt model khác để có thêm VRAM (app sẽ tự nạp lại với ngữ cảnh lớn hơn), "
                                   "hoặc chọn model không có chế độ suy nghĩ.")
            total = time.perf_counter() - t0
            yield ev(type="done", seconds=round(total, 1), tokens=tokens, tps=round(tokens / total, 1) if total > 0 else 0)
        except Exception as e:  # noqa: BLE001
            yield ev(type="error", text=str(e) or type(e).__name__)

    return StreamingResponse(gen(), media_type="application/x-ndjson", headers={"Cache-Control": "no-store"})

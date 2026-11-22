"""Phân tích hóa đơn bán lẻ: đọc ảnh hóa đơn bằng model thị giác, hiện bảng thống kê và ghi ra Excel.

  - Ảnh hóa đơn:   Invoice_Ana/Raw_invoice        (jpg, png, webp, bmp, tif, gif)
  - File mẫu:      Invoice_Ana/Template invoice   (file .xlsx đầu tiên; dòng 1 là tiêu đề cột)
  - Kết quả:       Invoice_Ana/Ket qua/Thong ke hoa don <ngày-giờ>.xlsx

Mỗi dòng hàng hóa trên hóa đơn thành 1 dòng trong bảng; thông tin chung của hóa đơn lặp lại ở từng dòng.
"""
from __future__ import annotations

import base64
import io
import json
import re
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from openpyxl import load_workbook
from PIL import Image, ImageOps

from ..core import ROOT, Context, Module, Param, ensure_loaded
from ..ollama import GenStats

BASE = ROOT / "Invoice_Ana"
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".gif"}
MAX_SIDE = 2048  # ảnh lớn hơn sẽ được thu nhỏ trước khi gửi cho model

# Thứ tự các trường khớp đúng thứ tự cột A→R của file mẫu.
INVOICE_FIELDS = {
    "invoice_no": "Số hóa đơn in ở góc trên bên phải (ví dụ 0062272)",
    "seller_name": "Đơn vị bán hàng (thường nằm trong con dấu đóng ở phần đầu)",
    "seller_address": "Địa chỉ đơn vị bán hàng",
    "seller_account": "Số tài khoản đơn vị bán hàng",
    "seller_phone": "Điện thoại đơn vị bán hàng",
    "seller_tax_code": "Mã số thuế (MST) của đơn vị bán hàng",
    "buyer_name": "Họ tên người mua hàng",
    "buyer_company": "Tên đơn vị mua hàng",
    "buyer_address": "Địa chỉ đơn vị mua hàng",
    "buyer_account": "Số tài khoản đơn vị mua hàng",
    "payment_method": "Hình thức thanh toán (ví dụ TM, CK)",
}
ITEM_FIELDS = {
    "no": "Số thứ tự",
    "name": "Tên hàng hóa, dịch vụ",
    "unit": "Đơn vị tính",
    "quantity": "Số lượng, chỉ ghi chữ số",
    "unit_price": "Đơn giá, chỉ ghi chữ số, không có dấu chấm phẩy (ví dụ 75000)",
    "amount": "Thành tiền, chỉ ghi chữ số (ví dụ 1050000)",
}
_STR = {"type": "string"}
SCHEMA = {
    "type": "object",
    "properties": {
        **{k: _STR for k in INVOICE_FIELDS},
        "items": {
            "type": "array",
            "items": {"type": "object", "properties": {k: _STR for k in ITEM_FIELDS}, "required": list(ITEM_FIELDS)},
        },
        "amount_in_words": _STR,
    },
    "required": [*INVOICE_FIELDS, "items", "amount_in_words"],
}
PROMPT = (
    "Đây là ảnh một hóa đơn bán hàng của Việt Nam, có cả chữ in và chữ viết tay. "
    "Hãy đọc thật kỹ và trích xuất thông tin thành JSON với các trường sau:\n"
    + "\n".join(f'- "{k}": {v}' for k, v in INVOICE_FIELDS.items())
    + '\n- "items": danh sách các dòng hàng hóa, mỗi dòng gồm:\n'
    + "\n".join(f'    - "{k}": {v}' for k, v in ITEM_FIELDS.items())
    + '\n- "amount_in_words": Số tiền viết bằng chữ\n\n'
    "Quy tắc: chép đúng nguyên văn trên hóa đơn, giữ dấu tiếng Việt. "
    "Mỗi thông tin chỉ điền vào đúng một trường, không chép lặp sang trường khác. "
    "Đọc từng chữ số của mã số thuế, điện thoại, số hóa đơn; không bỏ sót chữ số nào. "
    'Dãy số trong các ô "MS" là mã số thuế, không phải số tài khoản. '
    "Cột nào của dòng hàng không ghi gì thì để rỗng, không lấy số của cột khác điền vào. "
    'Trường nào hóa đơn để trống hoặc không đọc được thì ghi chuỗi rỗng "". Tuyệt đối không tự bịa. '
    "Chỉ trả về JSON."
)


def load_image(path: Path) -> str:
    """Đọc ảnh (kể cả webp), xoay đúng chiều, thu nhỏ nếu quá lớn, trả về data URL dạng PNG."""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        if max(im.size) > MAX_SIDE:
            im.thumbnail((MAX_SIDE, MAX_SIDE))
        buf = io.BytesIO()
        im.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def parse_json(text: str) -> dict[str, Any]:
    """Lấy object JSON từ câu trả lời của model (kể cả khi bị bọc trong ```json)."""
    try:
        data = json.loads(text)
    except ValueError:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            raise ValueError("Model không trả về JSON") from None
        data = json.loads(m.group(0))
    if not isinstance(data, dict):
        raise ValueError("JSON không đúng dạng")
    return data


LOOP_REPEATS = 6  # cùng 1 câu suy nghĩ xuất hiện từng này lần thì coi là model đang lặp vòng


def looping(st: GenStats) -> bool:
    """Model đang suy nghĩ lặp vòng? (cùng 1 dòng dài lặp lại nhiều lần). Chỉ xét phần suy nghĩ:
    trong JSON trả lời, các dòng hàng hóa giống nhau là chuyện bình thường."""
    lines = [ln.strip() for ln in st.reasoning.splitlines() if len(ln.strip()) > 15]
    return bool(lines) and Counter(lines).most_common(1)[0][1] >= LOOP_REPEATS


def to_number(v: Any) -> float | int | str:
    """'1.050.000 đ' → 1050000, '14' → 14, '1,5' → 1.5. Không đổi được thì giữ nguyên chuỗi."""
    if isinstance(v, (int, float)):
        return v
    s = re.sub(r"[^\d.,\-]", "", str(v or ""))
    if not s:
        return ""
    if re.fullmatch(r"-?\d{1,3}([.,]\d{3})+", s) or re.fullmatch(r"-?\d+", s):
        return int(re.sub(r"[.,]", "", s))
    try:
        return float(s.replace(".", "").replace(",", "."))
    except ValueError:
        return str(v).strip()


def to_rows(inv: dict[str, Any]) -> list[list[Any]]:
    """Đổi 1 hóa đơn thành các dòng theo thứ tự cột của file mẫu (mỗi dòng hàng hóa 1 dòng)."""
    def s(d: dict[str, Any], k: str) -> str:
        return str(d.get(k) or "").strip()

    head = [s(inv, k) for k in INVOICE_FIELDS]
    words = s(inv, "amount_in_words")
    items = [it for it in inv.get("items") or [] if isinstance(it, dict)] or [{}]
    return [
        head + [s(it, "no"), s(it, "name"), s(it, "unit"),
                to_number(it.get("quantity")), to_number(it.get("unit_price")), to_number(it.get("amount")), words]
        for it in items
    ]


def write_excel(template: Path, out_dir: Path, rows: list[list[Any]]) -> Path:
    """Chép file mẫu rồi ghi các dòng dữ liệu vào dưới dòng tiêu đề."""
    wb = load_workbook(template)
    ws = wb.worksheets[0]
    start = ws.max_row + 1
    for r, row in enumerate(rows, start):
        for c, value in enumerate(row, 1):
            cell = ws.cell(r, c, value if value != "" else None)
            if isinstance(value, (int, float)):
                cell.number_format = "#,##0.##" if isinstance(value, float) else "#,##0"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"Thong ke hoa don {datetime.now():%Y%m%d-%H%M%S}.xlsx"
    wb.save(out)
    return out


def template_columns(template: Path) -> list[str]:
    ws = load_workbook(template, read_only=True).worksheets[0]
    return [str(c.value or "") for c in next(ws.iter_rows(min_row=1, max_row=1))]


class InvoiceAnalysis(Module):
    id = "invoice_analysis"
    name = "Phân tích hóa đơn bán lẻ"
    description = "Đọc ảnh hóa đơn trong thư mục, hiện bảng thống kê và ghi ra file Excel theo mẫu."
    category = "app"
    params = [
        Param("model", "Model thị giác", "model", help="Model càng lớn đọc chữ viết tay càng tốt", options=["vlm"]),
        Param("input_dir", "Thư mục ảnh hóa đơn", "text", str(BASE / "Raw_invoice")),
        Param("template_dir", "Thư mục chứa file Excel mẫu", "text", str(BASE / "Template invoice")),
        Param("output_dir", "Thư mục lưu kết quả", "text", str(BASE / "Ket qua")),
        Param("limit", "Số lượng hóa đơn cần chạy", "int", 0, "0 = chạy tất cả ảnh trong thư mục (theo thứ tự tên file)",
              min=0, max=100000, presets=[10, 20, 30], unit="hóa đơn"),
        Param("max_tokens", "Giới hạn token mỗi hóa đơn", "int", 0,
              "0 = không giới hạn: model suy nghĩ và viết tới khi xong", min=0, max=1000000),
        Param("unload_others", "Gỡ các model khác trước khi nạp", "bool", True),
    ]

    async def run(self, ctx: Context, p: dict[str, Any]) -> dict[str, Any]:
        in_dir, tpl_dir, out_dir = Path(p["input_dir"]), Path(p["template_dir"]), Path(p["output_dir"])
        if not in_dir.is_dir():
            raise RuntimeError(f"Không có thư mục ảnh hóa đơn: {in_dir}")
        files = sorted(f for f in in_dir.iterdir() if f.suffix.lower() in IMAGE_EXT)
        if not files:
            raise RuntimeError(f"Không có ảnh hóa đơn nào trong {in_dir}")
        if p["limit"]:
            files = files[: p["limit"]]
        templates = sorted(f for f in tpl_dir.glob("*.xlsx") if not f.name.startswith("~$")) if tpl_dir.is_dir() else []
        if not templates:
            raise RuntimeError(f"Không tìm thấy file Excel mẫu (.xlsx) trong {tpl_dir}")
        template = templates[0]
        columns = template_columns(template)
        ctx.log(f"{len(files)} ảnh hóa đơn · file mẫu: {template.name}")

        model = p["model"]
        await ensure_loaded(ctx, model, p["unload_others"])
        ctx.progress(1 / (len(files) + 1))

        rows: list[list[Any]] = []
        gallery: list[dict[str, Any]] = []
        raw: list[dict[str, Any]] = []
        failed = 0
        seconds: list[float] = []
        energy: list[float] = []  # Wh GPU tiêu thụ cho từng hóa đơn
        for i, f in enumerate(files):
            card: dict[str, Any] = {"key": f.name, "title": f.name, "status": "pending"}
            try:
                card["src"] = load_image(f)
                ctx.show(card)
                mark = ctx.gpu.mark()
                t = time.perf_counter()
                async def ask(**extra: Any) -> GenStats:
                    return await ctx.lm.chat(
                        model,
                        [{"role": "user", "content": [{"type": "text", "text": PROMPT},
                                                      {"type": "image_url", "image_url": {"url": card["src"]}}]}],
                        max_tokens=p.get("max_tokens") or -1,  # -1 = Ollama không giới hạn (chỉ dừng khi đầy ngữ cảnh)
                        response_format={"type": "json_schema",
                                         "json_schema": {"name": "invoice", "strict": True, "schema": SCHEMA}},
                        should_stop=looping,
                        **extra,
                    )

                s = await ask()
                if s.finish_reason == "aborted":
                    ctx.log(f"[{i + 1}/{len(files)}] {f.name}: model suy nghĩ lặp vòng (sau {s.completion_tokens} token), "
                            "hỏi lại với chế độ suy nghĩ tắt")
                    s = await ask(reasoning_effort="none")
                # Model có "suy nghĩ" (gemma, qwen...) đôi khi nghĩ hết giới hạn token (hoặc đầy ngữ cảnh) mà chưa viết ra JSON.
                # Khi đó hỏi lại 1 lần với chế độ suy nghĩ tắt; thời gian và điện vẫn tính cả 2 lần hỏi.
                elif not s.text.strip() and s.reasoning:
                    ctx.log(f"[{i + 1}/{len(files)}] {f.name}: model suy nghĩ hết {s.completion_tokens} token "
                            "mà chưa trả lời, hỏi lại với chế độ suy nghĩ tắt")
                    s = await ask(reasoning_effort="none")
                seconds.append(time.perf_counter() - t)
                wh = ctx.gpu.energy_wh(mark)
                energy.append(wh or 0)
                if not s.text.strip():
                    raise ValueError("Model không trả lời (hết giới hạn token)" if s.finish_reason == "length"
                                     else "Model trả lời rỗng")
                if s.finish_reason == "aborted":
                    raise ValueError("Model lặp vòng, không trả lời xong")
                if s.finish_reason == "length":
                    raise ValueError("Câu trả lời bị cắt giữa chừng vì hết giới hạn token")
                inv = parse_json(s.text)
                inv_rows = to_rows(inv)
                rows += inv_rows
                total = sum(r[16] for r in inv_rows if isinstance(r[16], (int, float)))
                card = {**card, "status": "done",
                        "meta": f"{seconds[-1]:.1f} giây · {s.completion_tokens} token · {s.gen_tps:.0f} tok/s"
                                + (f" · {wh:.2f} Wh" if wh else ""),
                        "fields": [{"label": "Mã hóa đơn", "value": inv_rows[0][0]},
                                   {"label": "Đơn vị bán", "value": inv_rows[0][1]},
                                   {"label": "Đơn vị mua", "value": inv_rows[0][7]},
                                   {"label": "Số dòng hàng", "value": len(inv.get("items") or [])},
                                   {"label": "Thành tiền", "value": f"{total:,.0f}".replace(",", ".") + " đ"}]}
                raw.append({"file": f.name, "data": inv})
                ctx.log(f"[{i + 1}/{len(files)}] {f.name}: hóa đơn {inv_rows[0][0] or '?'}, "
                        f"{len(inv_rows)} dòng, {seconds[-1]:.1f} giây")
            except Exception as e:  # noqa: BLE001 - 1 ảnh lỗi không làm hỏng cả lô
                failed += 1
                card = {**card, "status": "done", "fields": [{"label": "Lỗi", "value": str(e)[:200]}]}
                raw.append({"file": f.name, "error": str(e)})
                ctx.log(f"[{i + 1}/{len(files)}] {f.name}: LỖI - {e}")
            ctx.show(card)
            gallery.append(card)
            ctx.progress((i + 2) / (len(files) + 1))

        result: dict[str, Any] = {
            "summary": [
                {"label": "Hóa đơn đã đọc", "value": len(files) - failed, "unit": f"/ {len(files)} ảnh"},
                {"label": "Số dòng hàng hóa", "value": len(rows), "unit": "dòng"},
                {"label": "Tổng thành tiền", "unit": "đ",
                 "value": sum(r[16] for r in rows if isinstance(r[16], (int, float)))},
                {"label": "Thời gian mỗi hóa đơn", "unit": "giây",
                 "value": round(sum(seconds) / len(seconds), 1) if seconds else None},
                *([{"label": "Điện mỗi hóa đơn", "unit": "Wh", "value": round(sum(energy) / len(energy), 3)}]
                  if sum(energy) else []),
            ],
            "images": gallery,
            "tables": [{"title": "Bảng thống kê hóa đơn (giống file Excel)", "columns": columns, "rows": rows}],
            "raw": {"invoices": raw, "template": str(template)},
        }
        if rows:
            out = write_excel(template, out_dir, rows)
            ctx.log(f"Đã ghi Excel: {out}")
            result["files"] = [{"label": "File Excel thống kê", "path": str(out)}]
        return result


MODULE = InvoiceAnalysis()

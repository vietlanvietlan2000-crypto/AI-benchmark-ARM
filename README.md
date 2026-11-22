# AI Bench Spark

Phiên bản của AI Bench dành cho **RTX Spark – Windows 11 on ARM**, chạy model AI bằng **Ollama**.
Gồm đủ các tính năng của AI Bench: benchmark GPU, benchmark phân tích hình ảnh, phân tích hóa đơn, và chatbot hỏi đáp
tài liệu (RAG: OCR, trích dẫn trang, tô vàng trên trang gốc, lịch sử hội thoại, xuất Excel, tìm kiếm internet).

Khác với AI Bench (LM Studio, Windows x64):

| | AI Bench | AI Bench Spark |
|---|---|---|
| Nền chạy model | LM Studio (lệnh `lms`) | **Ollama** (API gốc `/api/chat`, `/api/embed`, `/api/ps`) |
| Máy | Windows x64, card NVIDIA rời | **Windows 11 ARM64** (RTX Spark), vẫn chạy được trên x64 |
| Python | x64 | **ARM64 gốc** (trình cài tự chọn, không chạy giả lập) |
| Bộ nhớ GPU | VRAM riêng | **Bộ nhớ chung** CPU–GPU: app tính ~75% RAM cho model |
| Dò vị trí chữ trên trang scan | RapidOCR | **Windows OCR** có sẵn trong Windows (chạy gốc ARM64, ~0,2 giây/trang) |
| Model OCR | chandra-ocr-2 | `qwen3-vl:*-instruct` của Ollama |
| Cổng giao diện | 8765 | **8770** (chạy song song được với AI Bench) |

## Cài đặt

Mở **PowerShell** và dán lệnh sau:

```powershell
irm https://raw.githubusercontent.com/vietlanvietlan2000-crypto/AI-benchmark-ARM/main/install.ps1 | iex
```

| Bước | Việc làm |
|---|---|
| Tải về | Nhận biết CPU ARM64 / x64, chọn thư mục cài, tải mã nguồn, cài `uv`, **Python ARM64** và thư viện |
| 1. Phần cứng | CPU, GPU, bộ nhớ (riêng hay dùng chung), dung lượng trống từng ổ |
| 2. Phần mềm | Driver NVIDIA, **Ollama** (thiếu thì hỏi rồi cài bằng `winget`), mở server Ollama |
| 3. Nơi lưu model | Đổi thư mục model của Ollama (biến `OLLAMA_MODELS`) sang ổ khác khi ổ hiện tại sắp đầy |
| 4. Model | Gợi ý **bộ ba** model trả lời / lập chỉ mục / OCR hợp với bộ nhớ máy, tải bằng `ollama pull` |
| 5. Hoàn tất | Tạo lối tắt "AI Bench Spark" ngoài Desktop, mở ứng dụng |

Chạy lại lệnh trên để cập nhật bản mới (kết quả, tài liệu, chỉ mục và lịch sử chat được giữ nguyên).
Chạy lại riêng phần kiểm tra máy và tải model: `uv run python -m aibench setup` (thêm `--lang en` để dùng tiếng Anh).
Danh sách model gợi ý: `aibench/catalog.json`. Driver NVIDIA cho Windows on ARM cần tự cài tại https://www.nvidia.com/drivers.

**Windows on ARM – cần biết:**
- Mọi thư viện Python đều có bản ARM64 gốc (numpy, lxml, pillow, pypdfium2, pydantic…), trừ RapidOCR (thiếu `pyclipper`)
  nên bản này dùng **Windows OCR** thay thế. Windows OCR đọc theo ngôn ngữ đã cài trong Windows; thiếu gói tiếng Việt thì
  đọc mất dấu, app vẫn so khớp được vì so khớp bỏ dấu.
- **GPU:** Ollama phải hỗ trợ GPU NVIDIA trên Windows ARM64 thì model mới chạy bằng GPU. Thanh trạng thái hiện rõ
  "⚡ Chạy bằng GPU" hay "🐢 Chạy bằng CPU" (lấy từ `/api/ps` của Ollama), nên biết ngay sau lần chạy đầu tiên.
- **Bộ nhớ chung:** `nvidia-smi` trên GPU tích hợp không báo VRAM riêng; app lấy RAM hệ thống làm bộ nhớ GPU và chỉ tính
  ~75% cho model (phần còn lại cho Windows và ứng dụng khác).

## Chạy

- **Giao diện web:** bấm đúp `run.bat`, hoặc `uv run python -m aibench serve` rồi mở http://127.0.0.1:8770
- **Dòng lệnh:**

```
uv run python -m aibench models
uv run python -m aibench modules
uv run python -m aibench run gpu_benchmark -p model=gpt-oss:20b -p runs=3
uv run python -m aibench run vision_benchmark -p model=gemma3:12b
```

Nếu Ollama chưa chạy: bấm nút "Khởi động Ollama" trên giao diện, hoặc mở ứng dụng Ollama.
Tải model: `ollama pull <tên>` (danh sách tại https://ollama.com/library).

Kết quả mỗi lần chạy lưu thành file JSON trong `results/` và hiện ở mục Lịch sử.

**Ngữ cảnh khi đo:** benchmark nạp model với ngữ cảnh 8.192 token (`num_ctx`) để các lần đo so sánh được với nhau.
Bài "Song song ×N" chỉ chạy song song thật khi Ollama cho phép nhiều yêu cầu cùng lúc (biến môi trường `OLLAMA_NUM_PARALLEL`);
nếu không, các yêu cầu xếp hàng và tổng tốc độ không cao hơn chạy đơn.

## Điện năng tiêu thụ

Mọi chức năng đều đo lượng điện GPU tiêu thụ (Wh), tính bằng cách cộng dồn công suất × thời gian từ các mẫu
`nvidia-smi` lấy mỗi 0,5 giây:

| Chức năng | Chỉ số điện |
|---|---|
| Mọi module (tự động) | Điện GPU tiêu thụ cả lần chạy, kể cả lúc nạp model |
| Benchmark GPU | Điện từng kịch bản, điện mỗi 1.000 token |
| Benchmark hình ảnh | Điện từng độ phân giải, điện mỗi ảnh |
| Phân tích hóa đơn | Điện từng hóa đơn (ghi dưới mỗi ảnh), điện trung bình mỗi hóa đơn |

Thanh trên cùng của giao diện hiện công suất GPU tức thời (W). Module mới tự có chỉ số "cả lần chạy";
muốn đo riêng một đoạn thì dùng `mark = ctx.gpu.mark()` rồi `ctx.gpu.energy_wh(mark)`.

Đây chỉ là điện của card đồ họa, **không gồm CPU, màn hình và phần còn lại của máy**. 1.000 Wh = 1 kWh = 1 số điện.

## Benchmark GPU đo những gì

| Kịch bản | Mục đích |
|---|---|
| Sinh ngắn | Tốc độ sinh (tok/s) và độ trễ token đầu tiên (TTFT) |
| Ngữ cảnh dài | Tốc độ GPU đọc prompt ~2700 token |
| Sinh dài | Tốc độ khi chạy ổn định, kèm % GPU, nhiệt độ, công suất |
| Song song ×N | Tổng thông lượng khi nhiều yêu cầu chạy cùng lúc |

**Điểm tổng** = 10 × trung bình nhân của (tok/s sinh ngắn, tok/s sinh dài, tok/s song song, tok/s đọc prompt ÷ 10).
Điểm chỉ so sánh được giữa các lần chạy **cùng model, cùng tham số**.

Lưu ý khi đọc số:
- Tốc độ đọc prompt là xấp xỉ (số token prompt ÷ TTFT).
- Thời gian nạp model chỉ đo khi model chưa nạp sẵn lúc bấm Chạy.
- Với model "suy nghĩ" (gpt-oss, qwen), token suy nghĩ cũng được tính là token sinh ra.
- Model lớn hơn bộ nhớ GPU sẽ có một phần chạy bằng CPU và chậm hẳn — thanh trạng thái báo "Chạy bằng GPU" hay không.

## Benchmark phân tích hình ảnh đo những gì

Phần mềm tự vẽ ảnh test vuông ở nhiều độ phân giải (mặc định 448, 896, 1344, 2048 px) và gửi tới model thị giác.
Mỗi ảnh có nội dung ngẫu nhiên đã biết đáp án, nên chấm được cả tốc độ lẫn độ chính xác:

| Trong ảnh | Model phải làm | Kiểm tra |
|---|---|---|
| 3–8 hình tròn màu | Đếm số hình | Khả năng nhận biết vật thể |
| Mã lớn 6 ký tự | Đọc lại mã | Đọc chữ (OCR) |
| Dãy 5 chữ số nhỏ ở đáy ảnh | Đọc lại dãy số | Nhìn chi tiết nhỏ |

Kết quả gồm: số ảnh/phút, thời gian tới token đầu (gồm thời gian mã hoá ảnh), số token mỗi ảnh chiếm,
tỉ lệ đúng từng loại câu hỏi, VRAM, nhiệt độ, công suất.

**Điểm tổng** = số ảnh/phút × tỉ lệ trả lời đúng.

Trên giao diện, từng ảnh hiện ra ngay lúc AI đang xem, kèm câu trả lời của AI so với đáp án (✓/✗).
Các ảnh này được lưu cùng kết quả nên xem lại được trong Lịch sử; bấm vào ảnh để phóng to.
Module khác cũng dùng được cơ chế này: gọi `ctx.show(...)` khi đang chạy và trả về `images` trong kết quả.

Lưu ý: tốc độ phụ thuộc nhiều vào việc model có "suy nghĩ" trước khi trả lời hay không, nên chỉ so sánh
giữa các GPU khi dùng cùng model. Ảnh test là ảnh tổng hợp đơn giản, không đại diện cho ảnh y tế thật.

## Phân tích hóa đơn bán lẻ

| | Thư mục |
|---|---|
| Ảnh hóa đơn (jpg, png, webp, bmp, tif, gif) | `Invoice_Ana/Raw_invoice` |
| File Excel mẫu (dùng file `.xlsx` đầu tiên, dòng 1 là tiêu đề cột) | `Invoice_Ana/Template invoice` |
| File kết quả `Thong ke hoa don <ngày-giờ>.xlsx` | `Invoice_Ana/Ket qua` |

Chọn "Phân tích hóa đơn bán lẻ" trên giao diện rồi bấm Chạy. Ô "Số lượng hóa đơn cần chạy" giới hạn số ảnh
(0 = tất cả, theo thứ tự tên file); các nút "Chạy 10 / 20 / 30 hóa đơn" điền số và chạy luôn. Hoặc dùng dòng lệnh
(thêm `-p limit=10` để giới hạn):

```
uv run python -m aibench run invoice_analysis -p model=gemma3:27b
```

Model thị giác đọc từng ảnh và trả về JSON; phần mềm hiện bảng thống kê lên màn hình và ghi đúng bảng đó
vào bản sao của file mẫu. Mỗi dòng hàng hóa là 1 dòng; thông tin chung của hóa đơn lặp lại ở từng dòng.
Mỗi lần chạy tạo 1 file Excel mới, không ghi đè file cũ.

Thứ tự cột ghi ra cố định theo file mẫu hiện tại (A→R: mã hóa đơn, bên bán, bên mua, hình thức thanh toán,
dòng hàng, tổng tiền bằng chữ). Nếu đổi thứ tự cột trong file mẫu thì phải sửa `to_rows()` trong
`aibench/modules/invoice_analysis.py`.

**Phải kiểm tra lại kết quả.** Chữ in và con dấu đọc khá đúng, nhưng chữ viết tay vẫn sai đáng kể,
nhất là với model nhỏ. Ảnh chụp/scan càng nét thì càng đúng.

## Chatbot hỏi đáp tài liệu

Bấm **Chatbot** trên thanh đầu của giao diện.

1. Bỏ tài liệu vào thư mục `Tai_lieu/`, chia thư mục con tùy ý (ví dụ `Nhan su/`, `San pham/May X5/`).
   Đọc được: `.pdf` (có chữ hoặc bản scan), `.docx`, `.xlsx`, `.txt`, `.md`, `.csv` và file ảnh `.jpg`, `.png`, `.webp`, `.bmp`, `.tif`.
   Thư mục `Tai_lieu/Vi du/` là bộ tài liệu giả lập để thử, xóa đi khi dùng thật.
2. Chọn model lập chỉ mục (embedding) và model OCR, rồi bấm **Cập nhật chỉ mục**.
3. Chọn model trả lời và đặt câu hỏi.

**Lập chỉ mục gồm 5 bước** (giao diện hiện bước đang chạy, có nút **Dừng**):

| Bước | Việc làm |
|---|---|
| 1. Tiếp nhận | Quét thư mục; chỉ xử lý file mới hoặc đã sửa (đổi model embedding thì làm lại tất cả) |
| 2. Lấy chữ | PDF có chữ, Word, Excel, txt/md/csv: đọc thẳng. Trang PDF scan và file ảnh: OCR bằng model thị giác của Ollama |
| 3. Cắt đoạn | Chia thành đoạn khoảng 900 ký tự, ghi đoạn nằm ở trang / mục / sheet nào |
| 4. Vector hóa | Model embedding của Ollama đổi mỗi đoạn thành vector (`/api/embed`) |
| 5. Lưu trữ | Ghi vào cơ sở dữ liệu SQLite `rag_index/index.db`, từng tài liệu một |

- **OCR:** dùng model thị giác **bản `-instruct`**, ví dụ `qwen3-vl:8b-instruct` hoặc `qwen3-vl:2b-instruct` (máy ít bộ nhớ).
  Tag mặc định `qwen3-vl:2b` là bản *thinking*: nó chỉ suy nghĩ mà không chép chữ ra, app sẽ báo lỗi kèm hướng dẫn.
  Trong lúc OCR, các model khác được gỡ để nhường bộ nhớ; OCR xong thì gỡ model OCR rồi mới nạp model embedding.
- **Kết quả OCR được lưu lại** theo nội dung file: đổi model embedding hay lập lại chỉ mục không phải OCR lại.
- Chọn "Không OCR" thì trang scan bị bỏ qua; tài liệu đó hiện dấu ⚠ kèm số trang chưa OCR.
- Khi đang lập chỉ mục thì không chat và không chạy benchmark được (cùng dùng GPU).

**Trò chuyện tự nhiên.** Trước khi tra cứu, chatbot phân loại tin nhắn: câu chào hỏi, cảm ơn, hỏi "bạn làm được gì", hay yêu cầu
tóm tắt / dịch / lập bảng từ câu trả lời trước thì đáp lại ngay, không tra tài liệu (dưới câu trả lời ghi "💬 Trò chuyện").
Câu hỏi nối tiếp như "còn thai sản thì sao?" được viết lại thành câu tìm kiếm đầy đủ.

**Tìm bằng hai ngôn ngữ.** Câu tìm kiếm được dịch thêm sang tiếng Anh theo thuật ngữ kỹ thuật, giữ nguyên mã thiết bị và con số;
mỗi câu tự xếp hạng tài liệu rồi trộn thứ hạng, nên datasheet tiếng Anh không bị tài liệu tiếng Việt cùng chủ đề lấn át.

**Trích dẫn nguồn và tô vàng.** Mỗi ý trong câu trả lời có số nguồn `[2]`; bấm vào để mở khung xem tài liệu.
- **PDF và file ảnh** mở ở tab **📄 Trang gốc**: hiện đúng trang AI đã dùng, chỗ AI lấy thông tin được **tô vàng** trên trang.
  - PDF có chữ: vị trí lấy chính xác từ lớp chữ của PDF.
  - Trang scan / ảnh: **Windows OCR** dò vị trí từng dòng / ô chữ trên ảnh trang (khoảng 0,2 giây mỗi trang, lưu lại để lần sau
    mở ngay), rồi tô đúng các ô khớp câu trả lời. Dùng được với mọi model OCR của Ollama.
  - Tô theo độ phủ: dòng khớp nhất trước, rồi thêm các dòng mang thêm số liệu / từ mới của câu trả lời; không tô các hàng na ná nhau.
- Tab **📝 Văn bản** (và Word, Excel, Markdown, txt): chữ đã trích, đoạn được trích dẫn có nền vàng.

**Lịch sử trò chuyện.** Tự lưu sau mỗi câu trả lời vào `chat_history/chats.db` (không đưa lên GitHub). Cột "Lịch sử trò chuyện"
cho xem lại, chat tiếp, tìm trong tên lẫn nội dung, ✎ đổi tên, 🗑 xóa; mở lại app thì cuộc trò chuyện đang dở được mở lại.

**Bảng, Excel, biểu đồ.** "Lập bảng…" → bảng kèm nút **Xuất Excel**; "vẽ biểu đồ cột / đường / tròn…" → biểu đồ kèm
**Xuất Excel (kèm biểu đồ)**.

**Tìm kiếm qua internet.** Tick "Tìm kiếm qua internet" thì câu hỏi được gửi tới máy tìm kiếm (`ddgs`, không cần khóa API),
AI đọc vài trang kết quả đầu và trả lời kèm link nguồn. Mặc định tắt vì câu hỏi sẽ rời khỏi máy.

**Chọn model theo bộ nhớ.** Đầu thẻ Chatbot có ô **💡 Gợi ý cho máy này**: đề xuất bộ ba model trả lời / lập chỉ mục / OCR
trong các model đã tải, nút **Dùng gợi ý này** chọn cả ba. Mỗi model trong ô chọn có nhãn ✓ vừa máy / △ vừa sát / ✗ quá lớn.
Model trả lời và model lập chỉ mục chạy **cùng lúc** nên được tính chung bộ nhớ; model OCR chạy **một mình**.
Bảng tham khảo (tag mặc định trên thư viện Ollama, bản Q4; "bộ nhớ" = VRAM của card rời, hoặc ~75% RAM trên máy bộ nhớ chung):

| Bộ nhớ cho model | Model trả lời | Model lập chỉ mục | Model OCR |
|---|---|---|---|
| 8 GB | `gemma3:4b` (~3,3 GB) | `qwen3-embedding:0.6b` (~0,6 GB) | `qwen3-vl:2b-instruct` (~1,9 GB) |
| 16 GB | `gemma3:12b` (~8 GB) | `qwen3-embedding:0.6b` | `qwen3-vl:8b-instruct` (~6 GB) |
| 24 GB | `gpt-oss:20b` (~14 GB) | `qwen3-embedding:8b` (~4,7 GB) | `qwen3-vl:8b-instruct` |
| 32–48 GB | `gemma3:27b` (~17 GB) / `qwen3:30b` (~19 GB) | `qwen3-embedding:8b` | `qwen3-vl:8b-instruct` |
| 96 GB (máy 128 GB bộ nhớ chung) | `gpt-oss:120b` (~65 GB) | `qwen3-embedding:8b` | `qwen3-vl:8b-instruct` |

Dung lượng là xấp xỉ, có thể chênh tùy bản. Đổi model lập chỉ mục thì phải bấm "Cập nhật chỉ mục" một lần (không phải OCR lại).

Lưu ý:
- **Ngữ cảnh model trả lời** tự chọn theo bộ nhớ còn trống trong 65.536 / 32.768 / 16.384 / 8.192 token. Ollama không có lệnh
  ước lượng bộ nhớ như LM Studio nên app tính bằng công thức (dung lượng file + bộ nhớ ngữ cảnh), lệch về phía an toàn.
  App ghi nhớ ngữ cảnh đã chọn cho từng model và luôn gửi đúng con số đó (`num_ctx`), vì Ollama nạp lại model nếu ngữ cảnh đổi.
- Model có chế độ suy nghĩ (qwen3, gpt-oss…): khi trả lời từ tài liệu, phần suy nghĩ + trả lời giới hạn 8.192 token; chạm giới
  hạn hoặc hết ngữ cảnh mà chưa trả lời thì app tự hỏi lại với chế độ suy nghĩ tắt (`think: false`). App chỉ gửi `think` cho
  model có khả năng "thinking" (Ollama báo trong `/api/show`).
- Ô "Model lập chỉ mục" xếp model embedding lên đầu (Ollama báo khả năng "embedding"); model khác không dùng để lập chỉ mục được.
- Nút **Gỡ các model đã nạp** giải phóng bộ nhớ (Ollama `keep_alive: 0`); model cần dùng tự nạp lại ở câu hỏi kế tiếp.
  Model nạp qua app được giữ trong bộ nhớ 30 phút giữa các câu hỏi.
- AI vẫn có thể trả lời sai; tên tài liệu nguồn là để người dùng kiểm tra lại.

## Cấu trúc

```
aibench/
  core.py              Module, Param, Context, tự nạp module, chạy và lưu kết quả
  ollama.py            Gọi Ollama: hỏi đáp (stream, đo TTFT/tok/s), embedding, nạp/gỡ model, danh sách model
  engine.py            Model đang chạy bằng GPU / tràn sang CPU / chỉ CPU (Ollama /api/ps); RAM của máy
  gpu.py               Lấy mẫu GPU bằng nvidia-smi (%, bộ nhớ, nhiệt độ, công suất); bộ nhớ chung → RAM
  rag.py               Đọc file, OCR, chỉ mục, tìm kiếm hai ngôn ngữ
  pages.py             Vẽ trang gốc, Windows OCR dò vị trí chữ, chọn chỗ tô vàng
  chat.py              API chatbot; history.py  lịch sử hội thoại; websearch.py  tìm kiếm internet
  setup.py             Trình cài đặt từng bước (ARM64, Ollama, model)
  server.py            API + chạy job nền (FastAPI)
  web/index.html       Giao diện
  modules/             gpu_benchmark.py, vision_benchmark.py, invoice_analysis.py
Invoice_Ana/           Ảnh hóa đơn, file Excel mẫu, file kết quả
results/               Kết quả đã lưu
```

Cấu hình qua biến môi trường: `AIBENCH_OLLAMA_URL` (mặc định `http://127.0.0.1:11434`), `AIBENCH_GPU_INDEX` (mặc định `0`).

## Thêm tính năng mới

Tạo 1 file trong `aibench/modules/`, không cần sửa server hay giao diện:

```python
# aibench/modules/email_analysis.py
from ..core import Context, Module, Param

class EmailAnalysis(Module):
    id = "email_analysis"
    name = "Phân tích email"
    description = "Tóm tắt và phân loại một email."
    category = "app"
    params = [
        Param("model", "Model", "model"),
        Param("email", "Nội dung email", "textarea", ""),
    ]

    async def run(self, ctx: Context, p: dict) -> dict:
        s = await ctx.lm.chat(p["model"], [
            {"role": "system", "content": "Tóm tắt email, nêu mức độ khẩn cấp và việc cần làm."},
            {"role": "user", "content": p["email"]},
        ], max_tokens=1024)
        return {
            "summary": [{"label": "Tốc độ", "value": round(s.gen_tps, 1), "unit": "tok/s"}],
            "text": s.text,
        }

MODULE = EmailAnalysis()
```

Kiểu tham số hiện có: `model`, `int`, `bool`, `text`, `textarea`, `select`.
`run()` trả về `summary` (ô số liệu), `tables`, `charts`, `text` — giao diện tự hiển thị.

Với phân tích ảnh (X-quang): `ctx.lm.chat()` nhận `messages` chuẩn OpenAI nên gửi được ảnh dạng
`{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}` tới model thị giác
(các model ghi "thị giác" trong danh sách). Việc còn thiếu là thêm kiểu tham số `image` (ô tải ảnh) vào giao diện.

## Giấy phép

Miễn phí theo giấy phép [MIT](LICENSE): ai cũng được dùng, sửa và chia sẻ lại, kể cả cho mục đích thương mại, miễn là giữ dòng ghi bản quyền.

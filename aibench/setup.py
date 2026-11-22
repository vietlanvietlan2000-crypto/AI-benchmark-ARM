"""Trình cài đặt từng bước. Chạy: python -m aibench setup   (install.ps1 tự gọi sau khi tải mã nguồn)

  [1/5] Kiểm tra phần cứng      CPU ARM64 hay x64, GPU, bộ nhớ (riêng hay dùng chung), ổ đĩa
  [2/5] Kiểm tra phần mềm       driver NVIDIA, Ollama (thiếu thì hỏi rồi cài bằng winget), server Ollama
  [3/5] Nơi lưu model           đổi thư mục model của Ollama (biến OLLAMA_MODELS) khi ổ đĩa sắp đầy
  [4/5] Gợi ý và tải model      theo bộ nhớ của máy (danh sách ở catalog.json), tải bằng `ollama pull`
  [5/5] Hoàn tất                lối tắt ngoài Desktop, mở ứng dụng

Chỉ dùng thư viện chuẩn để chạy được ngay cả khi các gói khác chưa cài xong.
"""
from __future__ import annotations

import ctypes
import json
import os
import platform
import shutil
import string
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

from .engine import ram_total_mb
from .gpu import gpu_info
from .ollama import OLLAMA_URL, ollama_exe

ROOT = Path(__file__).resolve().parent.parent
OLLAMA_DEFAULT_MODELS = Path.home() / ".ollama" / "models"
WINGET_ID = "Ollama.Ollama"
UNIFIED_SHARE = 0.75  # máy bộ nhớ chung: phần RAM coi như GPU dùng được cho model (còn lại cho Windows, ứng dụng)
STEPS = 5
EN = False
YES = False


def L(vi: str, en: str) -> str:
    return en if EN else vi


def color(code: str, s: str) -> str:
    return f"\033[{code}m{s}\033[0m"


def step(n: int, title: str) -> None:
    done = "█" * n + "░" * (STEPS - n)
    print("\n" + color("96;1", f"[{n}/{STEPS}] {done}  {title}"))


def ok(s: str) -> None:
    print("  " + color("92", "✓") + " " + s)


def warn(s: str) -> None:
    print("  " + color("93", "!") + " " + s)


def bad(s: str) -> None:
    print("  " + color("91", "✗") + " " + s)


def info(s: str) -> None:
    print("    " + s)


def ask(question: str, default: str = "") -> str:
    """Hỏi người dùng; Enter (hoặc chế độ --yes, hoặc không có bàn phím) lấy giá trị mặc định."""
    if YES:
        print(f"  ? {question} → {default or '-'}")
        return default
    try:
        return input(color("97;1", f"  ? {question} ")).strip() or default
    except EOFError:
        print()
        return default


def yes(question: str, default: bool = True) -> bool:
    a = ask(question + (" [Y/n]" if default else " [y/N]"), "y" if default else "n").lower()
    return a[:1] in ("y", "c")  # y = yes, c = có


def run_live(cmd: list[str]) -> int:
    """Chạy lệnh và để nó in thẳng tiến độ ra màn hình."""
    try:
        return subprocess.run(cmd).returncode
    except OSError as e:
        bad(str(e))
        return 1


def gb(mb: float) -> str:
    return f"{mb / 1024:.1f}".replace(".", "," if not EN else ".")


# ----- phần cứng -------------------------------------------------------------------------------
def machine_arch() -> dict[str, str]:
    """Kiến trúc CPU thật của máy và của Python đang chạy. Trên Windows ARM, Python x64 vẫn chạy được nhờ giả lập
    nhưng chậm hơn, nên cần biết cả hai. Kết quả: {"machine": "ARM64"|"x64"|..., "python": "ARM64"|"x64"|...}."""
    names = {0xAA64: "ARM64", 0x8664: "x64", 0x014C: "x86"}
    machine = ""
    if sys.platform == "win32":
        try:  # IsWow64Process2 trả kiến trúc thật của máy, kể cả khi tiến trình này đang chạy giả lập
            proc, native = ctypes.c_ushort(), ctypes.c_ushort()
            if ctypes.windll.kernel32.IsWow64Process2(ctypes.windll.kernel32.GetCurrentProcess(),
                                                       ctypes.byref(proc), ctypes.byref(native)):
                machine = names.get(native.value, hex(native.value))
        except (AttributeError, OSError):
            pass
    py = {"arm64": "ARM64", "aarch64": "ARM64", "amd64": "x64", "x86_64": "x64"}.get(platform.machine().lower(), platform.machine())
    return {"machine": machine or py, "python": py}


def other_gpus() -> list[str]:
    """Tên các card đồ họa Windows thấy (dùng khi không có nvidia-smi)."""
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_VideoController).Name"],
                             capture_output=True, text=True, timeout=20).stdout
        return [x.strip() for x in out.splitlines() if x.strip()]
    except (OSError, subprocess.SubprocessError):
        return []


def drives() -> list[tuple[str, float, float]]:
    """[(ổ, GB trống, GB tổng)] của các ổ đĩa cố định."""
    out = []
    for letter in string.ascii_uppercase:
        root = f"{letter}:\\"
        if ctypes.windll.kernel32.GetDriveTypeW(root) != 3:  # 3 = ổ cố định
            continue
        try:
            u = shutil.disk_usage(root)
        except OSError:
            continue
        out.append((root, u.free / 2**30, u.total / 2**30))
    return out


def free_gb(path: Path) -> float:
    p = path
    while not p.exists() and p != p.parent:
        p = p.parent
    try:
        return shutil.disk_usage(p).free / 2**30
    except OSError:
        return 0.0


def model_memory_mb(g: dict[str, Any], ram_mb: float) -> float:
    """Bộ nhớ dành cho model: VRAM của card rời, hoặc ~75% RAM trên máy bộ nhớ chung (GPU tích hợp)."""
    if g.get("unified"):
        return (g.get("vram_total_mb") or ram_mb) * UNIFIED_SHARE
    return g.get("vram_total_mb") or 0


def check_hardware() -> dict[str, Any]:
    step(1, L("Kiểm tra phần cứng", "Checking hardware"))
    hw: dict[str, Any] = {"gpu": gpu_info(), "ram_mb": ram_total_mb() or 0, "arch": machine_arch()}
    a = hw["arch"]
    ok(L(f"CPU: {a['machine']} · {os.cpu_count()} luồng", f"CPU: {a['machine']} · {os.cpu_count()} threads"))
    if a["machine"] == "ARM64" and a["python"] != "ARM64":
        warn(L(f"Python đang chạy bản {a['python']} (giả lập) trên máy ARM64: vẫn dùng được nhưng chậm hơn. "
               "Chạy lại install.ps1 để cài Python ARM64.",
               f"Python is the {a['python']} build (emulated) on an ARM64 PC: it works but slower. Re-run install.ps1 to get ARM64 Python."))
    g = hw["gpu"]
    if g.get("name"):
        mem = (L(f"bộ nhớ chung với CPU ({gb(g['vram_total_mb'])} GB RAM, model dùng được khoảng {gb(model_memory_mb(g, hw['ram_mb']))} GB)",
                 f"memory shared with the CPU ({gb(g['vram_total_mb'])} GB RAM, about {gb(model_memory_mb(g, hw['ram_mb']))} GB usable for models)")
               if g.get("unified") else f"{gb(g['vram_total_mb'])} GB VRAM")
        ok(f"GPU: {g['name']} · {mem} · driver {g['driver']}")
    else:
        names = other_gpus()
        warn(L("Không thấy GPU NVIDIA (không có lệnh nvidia-smi).", "No NVIDIA GPU found (nvidia-smi is missing)."))
        if names:
            info(L("Card đồ họa trong máy: ", "Graphics adapters: ") + ", ".join(names))
        info(L("App vẫn chạy được nhưng không đo được nhiệt độ và điện năng GPU.",
               "The app still runs, but GPU temperature and power cannot be measured."))
    ok(f"RAM: {gb(hw['ram_mb'])} GB")
    for root, free, total in drives():
        (ok if free > 20 else warn)(L(f"Ổ {root} còn trống {free:.0f} GB / {total:.0f} GB", f"Drive {root} {free:.0f} GB free of {total:.0f} GB"))
    return hw


# ----- phần mềm --------------------------------------------------------------------------------
def server_up() -> bool:
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/version", timeout=3) as r:
            return r.status == 200
    except OSError:
        return False


def start_ollama() -> None:
    exe = ollama_exe()
    if not exe:
        return
    app = Path(exe).with_name("ollama app.exe")
    env = dict(os.environ)
    md = models_dir()
    if md != OLLAMA_DEFAULT_MODELS:
        env["OLLAMA_MODELS"] = str(md)  # tiến trình này có thể chưa thấy biến vừa ghi bằng setx
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "DETACHED_PROCESS", 0)
    cmd = [str(app)] if app.is_file() else [exe, "serve"]
    subprocess.Popen(cmd, env=env, creationflags=flags, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def stop_ollama() -> None:
    """Tắt hẳn Ollama (ứng dụng khay và server) để nó đọc lại cài đặt khi mở lại."""
    for name in ("ollama app.exe", "ollama.exe"):
        subprocess.run(["taskkill", "/F", "/T", "/IM", name], capture_output=True,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    time.sleep(2)


def wait_for(cond, seconds: int, label: str) -> bool:
    t0 = time.time()
    while time.time() - t0 < seconds:
        if cond():
            print()
            return True
        print(f"\r    {label} {int(time.time() - t0)}s / {seconds}s ", end="", flush=True)
        time.sleep(2)
    print()
    return False


def check_software() -> bool:
    """Trả True nếu Ollama dùng được (đã cài và server chạy)."""
    step(2, L("Kiểm tra phần mềm cần thiết", "Checking required software"))
    build = sys.getwindowsversion().build
    ok(f"Windows {11 if build >= 22000 else sys.getwindowsversion().major} (build {build}) · Python {sys.version.split()[0]} ({machine_arch()['python']})")
    (ok if shutil.which("uv") else warn)("uv " + (L("đã có", "found") if shutil.which("uv") else L("chưa có trong PATH (chạy lại install.ps1 để cài)", "not on PATH (re-run install.ps1)")))
    (ok if shutil.which("nvidia-smi") else warn)(L("Driver NVIDIA (nvidia-smi) ", "NVIDIA driver (nvidia-smi) ") + (L("đã có", "found") if shutil.which("nvidia-smi")
        else L("chưa có – cài driver NVIDIA bản cho Windows ARM tại https://www.nvidia.com/drivers", "missing – install the NVIDIA driver for Windows on ARM from https://www.nvidia.com/drivers")))

    if not ollama_exe():
        bad(L("Ollama chưa được cài.", "Ollama is not installed."))
        if not shutil.which("winget"):
            info(L("Máy không có winget nên không tự cài được. Tải Ollama tại https://ollama.com/download rồi chạy lại.",
                   "winget is unavailable, so it cannot be installed automatically. Get Ollama from https://ollama.com/download and run setup again."))
            return False
        if not yes(L("Cài Ollama bằng winget ngay bây giờ (khoảng 1 GB)?", "Install Ollama with winget now (about 1 GB)?")):
            return False
        code = run_live(["winget", "install", "-e", "--id", WINGET_ID, "--accept-package-agreements", "--accept-source-agreements"])
        if code != 0 or not ollama_exe():
            bad(L("Cài Ollama không thành công. Hãy cài tay tại https://ollama.com/download rồi chạy lại.",
                  "Ollama installation failed. Install it manually from https://ollama.com/download and run setup again."))
            return False
    ok(L("Ollama đã cài", "Ollama is installed") + f": {ollama_exe()}")

    if not server_up():
        info(L("Đang mở Ollama...", "Starting Ollama..."))
        start_ollama()
        wait_for(server_up, 40, L("Đang chờ server Ollama", "Waiting for the Ollama server"))
    if server_up():
        try:
            with urllib.request.urlopen(f"{OLLAMA_URL}/api/version", timeout=3) as r:
                ver = json.loads(r.read()).get("version", "")
        except (OSError, ValueError):
            ver = ""
        ok(L(f"Server Ollama {ver} đang chạy ({OLLAMA_URL})", f"Ollama server {ver} is running ({OLLAMA_URL})"))
        return True
    warn(L("Chưa bật được server Ollama. Có thể bật sau bằng nút \"Khởi động Ollama\" trong app.",
           "Could not start the Ollama server. You can start it later with the \"Start Ollama\" button in the app."))
    return True


# ----- nơi lưu model ---------------------------------------------------------------------------
def models_dir() -> Path:
    """Thư mục model của Ollama: biến OLLAMA_MODELS (tiến trình này hoặc đã lưu cho tài khoản Windows), mặc định ~/.ollama/models."""
    v = os.environ.get("OLLAMA_MODELS")
    if not v and sys.platform == "win32":
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
                v = winreg.QueryValueEx(k, "OLLAMA_MODELS")[0]
        except OSError:
            v = None
    return Path(v) if v else OLLAMA_DEFAULT_MODELS


def set_models_dir(path: Path) -> bool:
    """Ghi biến OLLAMA_MODELS cho tài khoản Windows (setx). Ollama đọc biến này khi khởi động."""
    try:
        path.mkdir(parents=True, exist_ok=True)
        r = subprocess.run(["setx", "OLLAMA_MODELS", str(path)], capture_output=True, text=True,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if r.returncode != 0:
            raise OSError(r.stderr or r.stdout)
        os.environ["OLLAMA_MODELS"] = str(path)
        return True
    except OSError as e:
        bad(str(e))
        return False


def count_models(folder: Path) -> int:
    """Số model trong thư mục Ollama (mỗi model là 1 file trong manifests/<registry>/<namespace>/<tên>/<tag>)."""
    try:
        return sum(1 for f in (folder / "manifests").glob("*/*/*/*") if f.is_file())
    except OSError:
        return 0


def choose_models_dir() -> Path:
    step(3, L("Nơi lưu model AI", "Where to store AI models"))
    cur = models_dir()
    ok(L(f"Hiện tại: {cur}  (ổ này còn trống {free_gb(cur):.0f} GB)", f"Current: {cur}  ({free_gb(cur):.0f} GB free on this drive)"))
    ds = drives()
    for i, (root, free, _total) in enumerate(ds, 1):
        info(f"{i}. {root}  {L('còn trống', 'free')} {free:.0f} GB")
    info(L("Model AI nặng từ 1 đến 65 GB mỗi cái. Nếu ổ hiện tại sắp đầy, hãy chọn ổ khác.",
           "AI models take 1 to 65 GB each. If the current drive is nearly full, pick another one."))
    a = ask(L("Enter = giữ nguyên · nhập số thứ tự ổ đĩa hoặc đường dẫn thư mục để đổi:",
              "Enter = keep · type a drive number or a folder path to change:"))
    if not a:
        return cur
    new = Path(ds[int(a) - 1][0]) / "Ollama Models" if a.isdigit() and 1 <= int(a) <= len(ds) else Path(a.strip('"'))
    if not new.is_absolute():
        bad(L("Đường dẫn không hợp lệ, giữ nguyên thư mục cũ.", "Invalid path; keeping the current folder."))
        return cur
    if new.resolve() == cur.resolve():
        return cur
    if set_models_dir(new):
        ok(L(f"Model mới sẽ lưu tại: {new}", f"New models will be stored in: {new}"))
        info(L("Đang khởi động lại Ollama để nhận thư mục mới...", "Restarting Ollama so it uses the new folder..."))
        stop_ollama()
        start_ollama()
        wait_for(server_up, 40, L("Đang chờ server Ollama", "Waiting for the Ollama server"))
        info(L("Các model đã tải trước đó vẫn nằm ở thư mục cũ; muốn dùng tiếp thì chép sang thư mục mới.",
               "Models downloaded earlier stay in the old folder; copy them over if you still want them."))
        return new
    return cur


# ----- gợi ý model -----------------------------------------------------------------------------
def fit(size_gb: float, mem_mb: float, ram_mb: float) -> str:
    """ok / tight / slow / no. Cùng công thức với giao diện: file model +5% +300 MB + ngữ cảnh 8.192 token."""
    need = size_gb * 1024 * 1.05 + 300 + size_gb * 2.3 * 8
    if mem_mb:
        if need <= mem_mb - 1024:
            return "ok"
        if need <= mem_mb:
            return "tight"
    if need <= (mem_mb + ram_mb * 0.5 if mem_mb else ram_mb * 0.6):
        return "slow"
    return "no"


def installed_models() -> set[str]:
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=10) as r:
            names = {m.get("name", "") for m in json.loads(r.read()).get("models", [])}
        return names | {n.removesuffix(":latest") for n in names}
    except (OSError, ValueError):
        return set()


def suggest_models(hw: dict[str, Any], ready: bool, mdir: Path) -> None:
    step(4, L("Gợi ý model AI theo cấu hình máy", "Suggested AI models for this PC"))
    mem, ram = model_memory_mb(hw["gpu"], hw["ram_mb"]), hw["ram_mb"]
    catalog = json.loads((Path(__file__).parent / "catalog.json").read_text(encoding="utf-8"))["models"]
    have = installed_models()
    for m in catalog:
        m["fit"], m["have"] = fit(m["size_gb"], mem, ram), m["key"] in have
    # Đề xuất 1 bộ: model trả lời (có đọc ảnh) lớn nhất còn vừa khi chạy cùng model lập chỉ mục, + model lập chỉ mục
    # + model OCR. Model trả lời và lập chỉ mục chạy cùng lúc nên tính chung bộ nhớ.
    embeds = sorted((m for m in catalog if m["kind"] == "embed" and m["fit"] == "ok"), key=lambda m: -m["size_gb"])
    emb = next((e for e in embeds if e.get("multilingual")), embeds[0] if embeds else None)
    left = mem - (emb["size_gb"] * 1024 * 1.1 if emb else 0)
    chats = [m for m in catalog if m["kind"] == "chat" and fit(m["size_gb"], left, 0) == "ok"]
    chat = max((m for m in chats if m["vision"]), key=lambda m: m["size_gb"], default=None) or max(chats, key=lambda m: m["size_gb"], default=None)
    ocr = next((m for m in catalog if m["kind"] == "ocr" and m["fit"] == "ok"), None)
    picks = [m for m in (chat, emb, ocr) if m]
    info(L(f"Bộ nhớ cho model: {gb(mem)} GB" + (" (bộ nhớ chung)" if hw["gpu"].get("unified") else " VRAM") + f" · RAM {gb(ram)} GB",
           f"Memory for models: {gb(mem)} GB" + (" (shared memory)" if hw["gpu"].get("unified") else " VRAM") + f" · {gb(ram)} GB RAM"))
    label = {"ok": color("92", L("✓ phù hợp", "✓ fits")), "tight": color("93", L("△ vừa sát", "△ barely fits")),
             "slow": color("93", L("⚠ chạy được nhưng chậm", "⚠ runs but slowly")), "no": color("91", L("✗ quá lớn", "✗ too large"))}
    kinds = {"chat": L("trả lời   ", "answering "), "embed": L("lập chỉ mục", "indexing  "), "ocr": L("OCR       ", "OCR       ")}
    print()
    for i, m in enumerate(catalog, 1):
        star = color("96;1", "★") if m in picks else " "
        extra = L(" + ảnh", " + vision") if m["vision"] and m["kind"] == "chat" else "       "
        print(f"   {star} {i:2}. {m['key']:24} {m['size_gb']:5.1f} GB  {kinds[m['kind']]}{extra}  {label[m['fit']]}"
              + (L("  (đã có)", "  (installed)") if m["have"] else ""))
    print()
    info(L("★ = bộ đề xuất cho máy này. Model trả lời \"+ ảnh\" chạy được cả 3 bài test; model khác chỉ chạy Benchmark GPU.",
           "★ = suggested set for this PC. \"+ vision\" answering models run all 3 tests; others run the GPU benchmark only."))
    info(L("Dung lượng là của bản mặc định trên thư viện Ollama (ollama.com/library), có thể lệch đôi chút.",
           "Sizes are for the default tag on the Ollama library (ollama.com/library) and may differ slightly."))
    if not ready:
        warn(L("Ollama chưa sẵn sàng nên bỏ qua bước tải model. Chạy lại: uv run python -m aibench setup",
               "Ollama is not ready, so model download is skipped. Re-run: uv run python -m aibench setup"))
        return
    default = ",".join(str(catalog.index(m) + 1) for m in picks if not m["have"])
    a = ask(L(f"Nhập số các model muốn tải, cách nhau dấu phẩy (Enter = {default or 'không tải'}, 0 = bỏ qua):",
              f"Type the numbers of the models to download, comma-separated (Enter = {default or 'none'}, 0 = skip):"), default)
    chosen = [catalog[int(x) - 1] for x in a.replace(" ", "").split(",") if x.isdigit() and 1 <= int(x) <= len(catalog)]
    chosen = list({m["key"]: m for m in chosen if not m["have"]}.values())  # 1 model có thể vừa trả lời vừa OCR
    if not chosen:
        info(L("Không tải model nào.", "No models to download."))
        return
    total, free = sum(m["size_gb"] for m in chosen), free_gb(mdir)
    risky = [m["key"] for m in chosen if m["fit"] in ("slow", "no")]
    if risky and not yes(L(f"{', '.join(risky)} không hợp cấu hình máy này (sẽ rất chậm hoặc không chạy được). Vẫn tải?",
                           f"{', '.join(risky)} does not suit this PC (very slow or will not run). Download anyway?"), False):
        chosen = [m for m in chosen if m["fit"] not in ("slow", "no")]
        total = sum(m["size_gb"] for m in chosen)
    if chosen and total + 2 > free:
        bad(L(f"Cần {total:.0f} GB nhưng {mdir} chỉ còn {free:.0f} GB. Chạy lại setup và chọn ổ khác ở bước 3.",
              f"{total:.0f} GB needed but {mdir} only has {free:.0f} GB free. Re-run setup and pick another drive in step 3."))
        return
    for i, m in enumerate(chosen, 1):
        print(color("96", f"\n  ↓ [{i}/{len(chosen)}] {m['key']} ({m['size_gb']:.1f} GB)  →  {mdir}"))
        code = run_live([ollama_exe() or "ollama", "pull", m["key"]])
        (ok if code == 0 else bad)(m["key"] + (L(" đã tải xong", " downloaded") if code == 0 else L(" tải lỗi (thử lại: ollama pull ", " failed (retry: ollama pull ") + m["key"] + ")"))


# ----- hoàn tất --------------------------------------------------------------------------------
def make_shortcut() -> bool:
    ps = ("$d=[Environment]::GetFolderPath('Desktop');$s=(New-Object -ComObject WScript.Shell).CreateShortcut(\"$d\\AI Bench Spark.lnk\");"
          f"$s.TargetPath='{ROOT / 'run.bat'}';$s.WorkingDirectory='{ROOT}';$s.IconLocation='shell32.dll,12';$s.Save()")
    try:
        return subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, timeout=30).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def finish() -> None:
    step(5, L("Hoàn tất", "Finishing"))
    if yes(L("Tạo lối tắt \"AI Bench Spark\" ngoài Desktop?", "Create an \"AI Bench Spark\" shortcut on the Desktop?")):
        (ok if make_shortcut() else warn)(L("Lối tắt ngoài Desktop", "Desktop shortcut"))
    ok(L(f"AI Bench Spark đã cài tại: {ROOT}", f"AI Bench Spark is installed in: {ROOT}"))
    info(L("Mở ứng dụng: bấm đúp run.bat (hoặc lối tắt AI Bench Spark), rồi vào http://127.0.0.1:8770",
           "To start: double-click run.bat (or the AI Bench Spark shortcut), then open http://127.0.0.1:8770"))
    info(L("Chạy lại trình cài đặt bất cứ lúc nào: uv run python -m aibench setup", "Re-run this setup any time: uv run python -m aibench setup"))
    if not YES and yes(L("Mở AI Bench Spark ngay bây giờ?", "Start AI Bench Spark now?")):
        os.startfile(ROOT / "run.bat")  # noqa: S606


def main(assume_yes: bool = False, lang: str | None = None) -> None:
    global EN, YES
    YES = assume_yes
    if sys.platform != "win32":
        sys.exit("AI Bench Spark setup supports Windows (ARM64 or x64) only.")
    os.system("")  # bật màu ANSI trên console Windows cũ
    print(color("96;1", "\n  AI BENCH SPARK · Tech by Lân") + "  –  setup (Ollama · Windows on ARM)")
    EN = (lang or ask("Ngôn ngữ / Language: [1] Tiếng Việt  [2] English", "1")).lower() in ("2", "en", "english")
    hw = check_hardware()
    ready = check_software()
    mdir = choose_models_dir() if ready else models_dir()
    if not ready:
        step(3, L("Nơi lưu model AI (bỏ qua vì Ollama chưa sẵn sàng)", "Model folder (skipped: Ollama is not ready)"))
    suggest_models(hw, ready, mdir)
    finish()

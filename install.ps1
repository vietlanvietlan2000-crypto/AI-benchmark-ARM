# AI Bench Spark installer (Windows 11 on ARM - RTX Spark; also works on Windows x64).
#
#   irm https://raw.githubusercontent.com/vietlanvietlan2000-crypto/AI-benchmark-ARM/main/install.ps1 | iex
#
# What it does / Viec no lam:
#   [1/4] check Windows and CPU (ARM64 / x64)     kiem tra Windows va CPU
#   [2/4] choose install folder                   chon thu muc cai dat
#   [3/4] download AI Bench Spark                 tai ma nguon
#   [4/4] install uv + Python (native ARM64 on ARM PCs) + libraries, then start the setup wizard
#         cai uv + Python (ban ARM64 tren may ARM) + thu vien, roi mo trinh cai dat tung buoc
#         (kiem tra phan cung, cai Ollama, chon noi luu model, goi y va tai model)
#
# Optional environment variables / Bien moi truong tuy chon:
#   AIBENCH_DIR     install folder (skips the question)
#   AIBENCH_SOURCE  local folder to copy from instead of downloading from GitHub
#   AIBENCH_NO_SETUP=1  do not start the setup wizard
#
# This file is kept ASCII-only on purpose: Windows PowerShell 5.1 misreads UTF-8 without BOM.

& {
    $ErrorActionPreference = 'Stop'
    $Repo   = 'vietlanvietlan2000-crypto/AI-benchmark-ARM'
    $Branch = 'main'
    $Total  = 4
    $PyVer  = '3.12'

    function Step($n, $text) { Write-Host ("`n[{0}/{1}] {2}" -f $n, $Total, $text) -ForegroundColor Cyan }
    function Ok($text)   { Write-Host "  [OK] $text" -ForegroundColor Green }
    function Note($text) { Write-Host "       $text" }
    function Warn($text) { Write-Host "  [!]  $text" -ForegroundColor Yellow }
    function Fail($text) { Write-Host "  [X]  $text" -ForegroundColor Red; throw $text }

    Write-Host "`n  AI BENCH SPARK - installer / trinh cai dat (Ollama - Windows on ARM)" -ForegroundColor Cyan
    try { [Console]::OutputEncoding = [Text.Encoding]::UTF8 } catch {}
    [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

    # ---- 1. Windows + CPU -------------------------------------------------------------------
    Step 1 'Checking Windows and CPU / Kiem tra Windows va CPU'
    if ($env:OS -ne 'Windows_NT') { Fail 'AI Bench Spark needs Windows 11 (ARM64 or x64). / Can Windows 11.' }
    if (-not [Environment]::Is64BitOperatingSystem) { Fail 'A 64-bit Windows is required. / Can Windows 64-bit.' }
    $build = [Environment]::OSVersion.Version.Build
    if ($build -lt 19041) { Fail "Windows build $build is too old. / Windows qua cu." }
    # Win32_Processor.Architecture: 12 = ARM64, 9 = x64. Reads the real CPU even if this PowerShell runs emulated.
    $arm = $false
    try { $arm = @(Get-CimInstance Win32_Processor | Where-Object { $_.Architecture -eq 12 }).Count -gt 0 } catch {}
    if (-not $arm -and ($env:PROCESSOR_ARCHITECTURE -eq 'ARM64' -or $env:PROCESSOR_ARCHITEW6432 -eq 'ARM64')) { $arm = $true }
    $cpu = if ($arm) { 'ARM64' } else { 'x64' }
    Ok "Windows build $build, CPU $cpu"
    if (-not $arm) { Warn 'This PC is not ARM64. AI Bench Spark still runs, but it is designed for RTX Spark (Windows on ARM). / May khong phai ARM64, van chay duoc.' }

    # ---- 2. Install folder ------------------------------------------------------------------
    Step 2 'Install folder / Thu muc cai dat'
    $default = Join-Path $env:LOCALAPPDATA 'AIBenchSpark'
    Get-PSDrive -PSProvider FileSystem | Where-Object { $_.Free } | ForEach-Object {
        Note ("{0}:\  free / con trong {1:N0} GB" -f $_.Name, ($_.Free / 1GB))
    }
    $dir = $env:AIBENCH_DIR
    if (-not $dir) {
        $dir = Read-Host "  ? Folder / Thu muc (Enter = $default)"
        if (-not $dir) { $dir = $default }
    }
    $dir = $dir.Trim('"')
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    $dir = (Resolve-Path $dir).Path
    Ok "Install to / Cai vao: $dir"

    # ---- 3. Download ------------------------------------------------------------------------
    Step 3 'Downloading AI Bench Spark / Tai ma nguon'
    if ($env:AIBENCH_SOURCE) {
        $src = (Resolve-Path $env:AIBENCH_SOURCE).Path
        Note "Copying from / Chep tu: $src"
    } else {
        $tmp = Join-Path ([IO.Path]::GetTempPath()) ("aibench-spark-" + [Guid]::NewGuid().ToString('N'))
        New-Item -ItemType Directory -Force -Path $tmp | Out-Null
        $zip = Join-Path $tmp 'src.zip'
        $url = "https://github.com/$Repo/archive/refs/heads/$Branch.zip"
        Note $url
        Invoke-WebRequest -Uri $url -OutFile $zip -UseBasicParsing
        Note 'Extracting / Dang giai nen...'
        Expand-Archive -Path $zip -DestinationPath $tmp -Force
        $src = (Get-ChildItem -Path $tmp -Directory | Select-Object -First 1).FullName
    }
    # Copy over the program files. Your results/, documents and chat history stay untouched (they are not in the repo).
    # Chep de len file chuong trinh. Ket qua, tai lieu va lich su chat cua ban duoc giu nguyen.
    robocopy $src $dir /E /NFL /NDL /NJH /NJS /NP /XD .git .venv .claude __pycache__ results rag_index chat_history | Out-Null
    if ($LASTEXITCODE -ge 8) { Fail "Copy failed (robocopy code $LASTEXITCODE). / Chep file loi." }
    if ($tmp) { Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue }
    if (-not (Test-Path (Join-Path $dir 'pyproject.toml'))) { Fail 'Download looks incomplete (pyproject.toml missing). / Tai chua du.' }
    Ok 'Source ready / Ma nguon da san sang'

    # ---- 4. uv + Python + libraries ---------------------------------------------------------
    Step 4 'Installing uv, Python and libraries / Cai uv, Python va thu vien'
    $uvBin = Join-Path $env:USERPROFILE '.local\bin'
    if (-not (Get-Command uv -ErrorAction SilentlyContinue) -and (Test-Path (Join-Path $uvBin 'uv.exe'))) { $env:Path = "$uvBin;$env:Path" }
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        Note 'uv is missing, installing it / Chua co uv, dang cai...'
        Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
        $env:Path = "$uvBin;$env:Path"
        if (-not (Get-Command uv -ErrorAction SilentlyContinue)) { Fail 'uv could not be installed. See https://docs.astral.sh/uv/ / Khong cai duoc uv.' }
    }
    Ok ("uv: " + (uv --version))
    # On ARM PCs use uv's native ARM64 Python, never an x64 Python that may already be installed (it would run emulated, slower).
    # May ARM: dung Python ARM64 cua uv, khong dung Python x64 co san (chay gia lap, cham hon).
    $py = if ($arm) { "cpython-$PyVer-windows-aarch64-none" } else { "cpython-$PyVer-windows-x86_64-none" }
    $env:UV_PYTHON = $py
    $env:UV_PYTHON_PREFERENCE = 'only-managed'
    Note "Downloading Python $PyVer ($cpu) and libraries (1-3 minutes) / Dang tai Python va thu vien (1-3 phut)..."
    uv python install $py
    if ($LASTEXITCODE -ne 0) { Fail "Could not install Python $py. / Khong cai duoc Python." }
    uv sync --directory $dir --python $py
    if ($LASTEXITCODE -ne 0) { Fail 'uv sync failed. / Cai thu vien loi.' }
    Ok "Python $PyVer ($cpu) and libraries installed / Da cai Python va thu vien"
    if ($env:AIBENCH_NO_SETUP -ne '1') {
        uv run --directory $dir python -m aibench setup
    }

    Write-Host "`n  Done / Xong.  Start / Mo ung dung:  $dir\run.bat`n" -ForegroundColor Green
}

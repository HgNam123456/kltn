# Khởi động llama-server (llama.cpp, backend Vulkan) trên AMD Radeon 780M.
# API OpenAI-compatible: http://localhost:8080/v1
#
# Dùng:  .\scripts\llm_server.ps1                                   (mặc định Gemma 4 E4B QAT + MTP)
#        .\scripts\llm_server.ps1 -Model models\Qwen3.5-2B-Q8_0.gguf -Draft ""   (không speculative)
#        .\scripts\llm_server.ps1 -Parallel 1 -CtxPerSlot 16384
#
# Tối ưu GPU đang bật:
#   --flash-attn on                 flash attention (KV cache gọn, prompt processing nhanh hơn)
#   --spec-type draft-mtp           speculative decoding bằng đầu MTP (multi-token prediction) của Gemma 4;
#   --model-draft mtp-*.gguf        với model khác: -SpecType ngram-simple (không cần model nháp) hoặc none
#   --parallel N --cont-batching    N slot, continuous batching: runner gửi N tin song song (run_eval --workers N)
#   --kv-unified-per-slot C         mỗi slot C token ctx, KV pool = N*C
#
# Sau đó đặt biến môi trường cho kgu:
#   $env:KGU_LLM_BASE_URL = "http://localhost:8080/v1"
#   $env:KGU_LLM_MODEL    = "local"
#   $env:KGU_LLM_API_KEY  = "none"

param(
    [string]$Model = "models\gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf",
    [string]$Draft = "models\mtp-gemma-4-E4B-it.gguf",
    [string]$SpecType = "draft-mtp",
    [int]$DraftMax = 2,
    [int]$Port = 8080,
    [int]$Parallel = 4,
    [int]$CtxPerSlot = 8192,
    [int]$GpuLayers = 99
)

$root = Split-Path -Parent $PSScriptRoot
$exe = Join-Path $root "tools\llama.cpp\llama-server.exe"
$modelPath = Join-Path $root $Model

if (-not (Test-Path $exe)) { Write-Error "Không thấy $exe. Tải llama-bXXXX-bin-win-vulkan-x64.zip và giải nén vào tools\llama.cpp\"; exit 1 }
if (-not (Test-Path $modelPath)) { Write-Error "Không thấy model $modelPath"; exit 1 }

$args = @(
    "--model", $modelPath,
    "--alias", "local",
    "--host", "127.0.0.1", "--port", $Port,
    "--n-gpu-layers", $GpuLayers,
    "--parallel", $Parallel, "--cont-batching",
    "--kv-unified-per-slot", $CtxPerSlot,
    "--jinja",
    "--reasoning", "off",
    "--temp", "0",
    "--flash-attn", "on"
)
if ($SpecType -and $SpecType -ne "none") {
    $args += @("--spec-type", $SpecType, "--spec-draft-n-max", $DraftMax)
    if ($Draft) {
        $draftPath = Join-Path $root $Draft
        if (-not (Test-Path $draftPath)) { Write-Error "Không thấy model nháp $draftPath"; exit 1 }
        $args += @("--model-draft", $draftPath)
    }
}

Write-Host "llama-server  model=$Model  draft=$Draft  spec=$SpecType  port=$Port  slots=$Parallel x $CtxPerSlot  ngl=$GpuLayers"
& $exe @args

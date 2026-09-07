# Khởi động llama-server (llama.cpp, backend Vulkan) trên AMD Radeon 780M.
# API OpenAI-compatible: http://localhost:8080/v1
#
# Dùng:  .\scripts\llm_server.ps1                       (mặc định Qwen3.5-2B Q8_0)
#        .\scripts\llm_server.ps1 -Model models\x.gguf -Port 8080 -Ctx 8192
#
# Sau đó đặt biến môi trường cho kgu:
#   $env:KGU_LLM_BASE_URL = "http://localhost:8080/v1"
#   $env:KGU_LLM_MODEL    = "local"
#   $env:KGU_LLM_API_KEY  = "none"

param(
    [string]$Model = "models\Qwen3.5-2B-Q8_0.gguf",
    [int]$Port = 8080,
    [int]$Ctx = 8192,
    [int]$Parallel = 1,
    [int]$GpuLayers = 99
)

$root = Split-Path -Parent $PSScriptRoot
$exe = Join-Path $root "tools\llama.cpp\llama-server.exe"
$modelPath = Join-Path $root $Model

if (-not (Test-Path $exe)) { Write-Error "Không thấy $exe. Tải llama-bXXXX-bin-win-vulkan-x64.zip và giải nén vào tools\llama.cpp\"; exit 1 }
if (-not (Test-Path $modelPath)) { Write-Error "Không thấy model $modelPath"; exit 1 }

Write-Host "llama-server  model=$Model  port=$Port  ctx=$Ctx  ngl=$GpuLayers"
& $exe `
    --model $modelPath `
    --alias local `
    --host 127.0.0.1 --port $Port `
    --ctx-size $Ctx `
    --n-gpu-layers $GpuLayers `
    --parallel $Parallel `
    --jinja `
    --reasoning off `
    --chat-template-kwargs '{\"enable_thinking\":false}' `
    --temp 0 `
    --flash-attn on

# Chạy kaggle/night/watch.sh tách khỏi phiên Claude/VS Code (khởi động qua WMI, không thuộc cây tiến trình nào) và giữ
# máy không tự ngủ (SetThreadExecutionState, tự hết hiệu lực khi tiến trình kết thúc). Log: results/kaggle/night/watch.log.
# Khởi động:
#   Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine =
#     'powershell.exe -NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File C:\Users\ADMIN\Desktop\kltn\kaggle\night\watch.ps1'}
# Dừng: Stop-Process -Id (Get-Content C:\Users\ADMIN\Desktop\kltn\results\kaggle\night\watch.pid)
$kl = "C:\Users\ADMIN\Desktop\kltn"
$dir = "$kl\results\kaggle\night"
New-Item -ItemType Directory -Force $dir | Out-Null
$PID | Out-File -Encoding ascii "$dir\watch.pid"
$env:KAGGLE_API_TOKEN = [Environment]::GetEnvironmentVariable("KAGGLE_API_TOKEN", "User")
Add-Type -Name Power -Namespace Win32 -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
[Win32.Power]::SetThreadExecutionState([uint32]2147483649) | Out-Null      # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
& "C:\Program Files\Git\bin\bash.exe" -c "bash /c/Users/ADMIN/Desktop/kltn/kaggle/night/watch.sh all >> /c/Users/ADMIN/Desktop/kltn/results/kaggle/night/watch.log 2>&1"

# Chạy kaggle/night/watch.sh tách khỏi phiên Claude/VS Code và giữ máy không tự ngủ trong lúc chạy
# (SetThreadExecutionState, tự hết hiệu lực khi tiến trình kết thúc). Log: results/kaggle/night/watch.log.
# Đêm 28/10: bản khởi động qua WMI (Invoke-CimMethod Win32_Process Create) chết lặng lẽ trước khi B xong (~23:1x, log
# trống) → hai làn được đẩy tay lúc 23:17; phần tải output sáng 29/10 chạy bằng Task Scheduler "kgu-night-fetch"
# (watch.ps1 fetch lúc 06:00, WakeToRun + StartWhenAvailable). Xóa tác vụ: Unregister-ScheduledTask kgu-night-fetch
param([string]$Mode = "all")
$dir = "C:\Users\ADMIN\Desktop\kltn\results\kaggle\night"
New-Item -ItemType Directory -Force $dir | Out-Null
$PID | Out-File -Encoding ascii "$dir\watch.pid"
$env:KAGGLE_API_TOKEN = [Environment]::GetEnvironmentVariable("KAGGLE_API_TOKEN", "User")
Add-Type -Name Power -Namespace Win32 -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
[Win32.Power]::SetThreadExecutionState([uint32]2147483649) | Out-Null      # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
& "C:\Program Files\Git\bin\bash.exe" -c "bash /c/Users/ADMIN/Desktop/kltn/kaggle/night/watch.sh $Mode >> /c/Users/ADMIN/Desktop/kltn/results/kaggle/night/watch.log 2>&1"

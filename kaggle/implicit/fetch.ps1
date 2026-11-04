# Task Scheduler "kgu-implicit-fetch": tải kết quả hai kernel kgu-implicit-* (kaggle/implicit/fetch.sh).
$dir = "C:\Users\ADMIN\Desktop\kltn\results\kaggle\implicit"
New-Item -ItemType Directory -Force $dir | Out-Null
$env:KAGGLE_API_TOKEN = [Environment]::GetEnvironmentVariable("KAGGLE_API_TOKEN", "User")
& "C:\Program Files\Git\bin\bash.exe" -c "bash /c/Users/ADMIN/Desktop/kltn/kaggle/implicit/fetch.sh >> /c/Users/ADMIN/Desktop/kltn/results/kaggle/implicit/fetch.log 2>&1"

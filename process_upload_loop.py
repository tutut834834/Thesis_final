import subprocess
import time
from datetime import datetime

print("PROCESS UPLOAD LOOP STARTED")
print("Uploads output_logs, descriptions, code_dump, and python/txt files every 60 seconds.")

while True:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    cmds = [
        "git pull --rebase origin main",
        "git add output_logs code_dump *.py *.txt PC*.txt",
        f'git commit -m "process upload loop {stamp}"',
        "git push"
    ]

    for cmd in cmds:
        print("RUN:", cmd)
        result = subprocess.run(cmd, shell=True)
        print("RETURN:", result.returncode)

    print("SLEEP 60 SEC")
    time.sleep(60)
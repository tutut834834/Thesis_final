import os
import time
import glob
import shutil
import subprocess
import datetime

PC_NAME = "PC03_H2_stealth_dirty"
BRANCH_NAME = "pc03-h2-stealth-dirty"

os.makedirs(PC_NAME, exist_ok=True)
os.makedirs(os.path.join(PC_NAME, "output_logs"), exist_ok=True)

print("PC03 FORCE LIVE UPLOADER STARTED")

while True:
    try:
        lock_path = os.path.join(".git", "index.lock")
        if os.path.exists(lock_path):
            print("Removing stale index.lock")
            try:
                os.remove(lock_path)
            except Exception as e:
                print("Could not remove lock:", e)
                time.sleep(20)
                continue

        stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(os.path.join(PC_NAME, "PC03_UPLOAD_HEARTBEAT.txt"), "w", encoding="utf-8") as f:
            f.write("PC03 live uploader heartbeat\n")
            f.write(stamp + "\n")
            f.write("Branch: pc03-h2-stealth-dirty\n")
            f.write("Experiment: H2 stealth dirty-label low-poison\n")

        for log_file in glob.glob(os.path.join("output_logs", "*PC03_H2_stealth_dirty*seed3*console_output.txt")):
            shutil.copy2(log_file, os.path.join(PC_NAME, "output_logs", os.path.basename(log_file)))

        for fname in ["federated.py", "agent.py", "aggregation.py", "models.py", "options.py", "utils.py"]:
            if os.path.exists(fname):
                shutil.copy2(fname, os.path.join(PC_NAME, fname))

        subprocess.run(f"git checkout {BRANCH_NAME}", shell=True)
        subprocess.run(f"git add -f output_logs {PC_NAME} *.py *.txt PC*.txt", shell=True)
        subprocess.run(f'git commit -m "PC03 force live upload {stamp}"', shell=True)
        subprocess.run(f"git push origin HEAD:{BRANCH_NAME}", shell=True)

        print("PC03_FORCE_UPLOAD_DONE", stamp)

    except Exception as e:
        print("PC03_FORCE_UPLOAD_FAILED", e)

    time.sleep(60)
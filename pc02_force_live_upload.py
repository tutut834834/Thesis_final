import os
import time
import glob
import shutil
import subprocess
import datetime

PC_NAME = "PC02_H1_clean"
BRANCH_NAME = "pc02-h1-clean"

os.makedirs(PC_NAME, exist_ok=True)
os.makedirs(os.path.join(PC_NAME, "output_logs"), exist_ok=True)

print("PC02 FORCE LIVE UPLOADER STARTED")

while True:
    try:
        if os.path.exists(os.path.join(".git", "index.lock")):
            print("Removing stale index.lock")
            try:
                os.remove(os.path.join(".git", "index.lock"))
            except Exception as e:
                print("Could not remove lock:", e)
                time.sleep(20)
                continue

        stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # heartbeat changes every time, forcing a commit
        with open(os.path.join(PC_NAME, "PC02_UPLOAD_HEARTBEAT.txt"), "w", encoding="utf-8") as f:
            f.write("PC02 live uploader heartbeat\n")
            f.write(stamp + "\n")
            f.write("Branch: pc02-h1-clean\n")
            f.write("Experiment: H1 clean-label non-IID\n")

        # copy current clean-label txt into visible PC02 folder
        for log_file in glob.glob(os.path.join("output_logs", "*cl1_cltype1_seed2*console_output.txt")):
            shutil.copy2(log_file, os.path.join(PC_NAME, "output_logs", os.path.basename(log_file)))

        # copy code too, for visible folder
        for fname in ["federated.py", "agent.py", "aggregation.py", "models.py", "options.py", "utils.py"]:
            if os.path.exists(fname):
                shutil.copy2(fname, os.path.join(PC_NAME, fname))

        subprocess.run(f"git checkout {BRANCH_NAME}", shell=True)
        subprocess.run(f"git add -f output_logs {PC_NAME} *.py *.txt PC*.txt", shell=True)
        subprocess.run(f'git commit -m "PC02 force live upload {stamp}"', shell=True)
        subprocess.run(f"git push origin HEAD:{BRANCH_NAME}", shell=True)

        print("PC02_FORCE_UPLOAD_DONE", stamp)

    except Exception as e:
        print("PC02_FORCE_UPLOAD_FAILED", e)

    time.sleep(60)
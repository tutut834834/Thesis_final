import os, time, glob, shutil, subprocess, datetime

BRANCH = "pc12-h6-privacy-medium-clean"
PC = "PC12_H6_privacy_medium_clean"

os.makedirs(PC, exist_ok=True)
os.makedirs(os.path.join(PC, "output_logs"), exist_ok=True)

while True:
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        lock = os.path.join(".git", "index.lock")
        if os.path.exists(lock):
            try:
                os.remove(lock)
            except:
                pass

        for f in glob.glob(os.path.join("output_logs", "*console_output*.txt")):
            shutil.copy2(f, os.path.join(PC, "output_logs", os.path.basename(f)))

        with open(os.path.join(PC, "PC12_HEARTBEAT.txt"), "w", encoding="utf-8") as h:
            h.write("PC12 H6 privacy medium clean alive\n" + stamp + "\n")

        subprocess.run(f"git checkout {BRANCH}", shell=True)
        subprocess.run(f"git add -f output_logs {PC} *.py *.txt", shell=True)
        subprocess.run(f'git commit -m "PC12 force upload {stamp}"', shell=True)
        subprocess.run(f"git push origin HEAD:{BRANCH}", shell=True)

        print("PC12_FORCE_UPLOAD_DONE", stamp)
    except Exception as e:
        print("PC12_FORCE_UPLOAD_FAILED", e)

    time.sleep(60)
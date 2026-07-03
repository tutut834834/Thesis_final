import subprocess
from datetime import datetime

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

def run(cmd):
    subprocess.run(cmd, shell=True)

run("git add output_logs logs code_dump *.py *.txt")
run(f'git commit -m "PC01 auto upload {stamp}"')
run("git push")
from pathlib import Path
from datetime import datetime

OUT = Path("CODE_DUMP_FINAL_CUCKOO_5NN.txt")

files = [
    "agent.py",
    "aggregation.py",
    "federated.py",
    "models.py",
    "options.py",
    "utils.py",
    "cuckoo_framework.py",
    "cuckoo_classifier_nest.py",
    "cuckoo_clean_label_egg.py",
    "cuckoo_multi_cuckoo.py",
    "run_final_cuckoo_5nn_dirty_100_sequential.slurm",
    "run_final_cuckoo_5nn_clean_pgd_bigsquare_100_sequential.slurm",
    "README_FINAL_CUCKOO_5NN.md",
    "GITHUB_PUSH_COMMANDS.txt",
]

with OUT.open("w", encoding="utf-8", errors="replace") as out:
    out.write("=" * 100 + "\n")
    out.write("CODE DUMP FOR FINAL CUCKOO 5NN PACKAGE\n")
    out.write(f"Created: {datetime.now()}\n")
    out.write("=" * 100 + "\n\n")

    for name in files:
        path = Path(name)
        out.write("\n\n" + "=" * 100 + "\n")
        out.write(f"FILE: {name}\n")
        out.write("=" * 100 + "\n\n")

        if path.exists():
            out.write(path.read_text(encoding="utf-8", errors="replace"))
        else:
            out.write(f"WARNING: {name} not found.\n")

        out.write("\n")

print("DONE")
print("Created:", OUT.resolve())
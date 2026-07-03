import os
from datetime import datetime

FILES_TO_DUMP = [
    "agent.py",
    "aggregation.py",
    "federated.py",
    "models.py",
    "options.py",
    "utils.py",
    "runner.sh",
]

OUTPUT_FOLDER = "code_dump"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = os.path.join(OUTPUT_FOLDER, f"all_code_files_{timestamp}.txt")

with open(output_file, "w", encoding="utf-8", errors="replace") as out:
    out.write("=" * 100 + "\n")
    out.write("CODE DUMP FOR THESIS PROJECT\n")
    out.write(f"Created: {datetime.now()}\n")
    out.write("=" * 100 + "\n\n")

    for filename in FILES_TO_DUMP:
        out.write("\n\n")
        out.write("=" * 100 + "\n")
        out.write(f"FILE: {filename}\n")
        out.write("=" * 100 + "\n\n")

        if not os.path.exists(filename):
            out.write(f"WARNING: {filename} not found.\n")
            continue

        try:
            with open(filename, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
                out.write(content)
        except Exception as e:
            out.write(f"ERROR reading {filename}: {e}\n")

print(f"Done. Code dump saved to:")
print(output_file)
import os
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

OUTPUT_DIR = os.path.join(BASE_DIR, "code_dump")
os.makedirs(OUTPUT_DIR, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = os.path.join(OUTPUT_DIR, f"full_code_dump_{timestamp}.txt")


def write_line(f, text=""):
    f.write(text + "\n")


def dump_file(path, f):
    try:
        write_line(f, "=" * 80)
        write_line(f, f"FILE: {path}")
        write_line(f, "=" * 80)

        with open(path, "r", encoding="utf-8", errors="ignore") as file:
            content = file.read()

        write_line(f, content)
        write_line(f, "\n")

    except Exception as e:
        write_line(f, f"[ERROR reading {path}]: {e}")


def main():
    with open(output_file, "w", encoding="utf-8") as f:

        write_line(f, "CUCKOO FRAMEWORK FULL CODE DUMP")
        write_line(f, f"Generated: {timestamp}")
        write_line(f, f"Base dir: {BASE_DIR}")
        write_line(f, "\n")

        for root, dirs, files in os.walk(BASE_DIR):

            # skip heavy folders
            if "__pycache__" in root or "output_logs" in root or "logs" in root:
                continue

            for file in files:
                if file.endswith(".py") or file.endswith(".bat") or file.endswith(".txt"):
                    full_path = os.path.join(root, file)
                    dump_file(full_path, f)

    print("Done. Code dump saved to:")
    print(output_file)


if __name__ == "__main__":
    main()
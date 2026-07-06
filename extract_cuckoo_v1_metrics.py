import csv
import glob
import os
import re
import sys


def fnum(x):
    try: return float(x)
    except Exception: return None


def parse_file(path):
    rows = []
    current_round = None
    val_acc = poison_acc = cum_poison = None
    cuckoo = {}
    mode = None
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if line.startswith('attack_mode='):
                mode = line.split('=', 1)[1]
            m = re.search(r'\| Round: (\d+) \|', line)
            if m:
                current_round = int(m.group(1))
            m = re.search(r'Val_Loss/Val_Acc: ([0-9.\-eE]+) / ([0-9.\-eE]+)', line)
            if m: val_acc = float(m.group(2))
            m = re.search(r'Poison Loss/Poison Acc: ([0-9.\-eE]+) / ([0-9.\-eE]+)', line)
            if m: poison_acc = float(m.group(2))
            m = re.search(r'Cumulative Poison Acc Mean: ([0-9.\-eE]+)', line)
            if m: cum_poison = float(m.group(1))
            if '=' in line and line.split('=', 1)[0] in {'lambda_t','host_norm','poison_norm','egg_raw_norm','egg_selected_norm','cuckoo_norm','cosine_host_cuckoo','sign_agreement_host_cuckoo','kept_fraction','classifier_kept_fraction','target_row_kept_fraction','base_row_kept_fraction','memory_norm'}:
                k, v = line.split('=', 1)
                cuckoo[k] = fnum(v)
            if line == 'CUCKOO_V1_BIOLOGY_NEST_END':
                pass
            if current_round is not None and val_acc is not None and poison_acc is not None:
                row = {'file': os.path.basename(path), 'round': current_round, 'attack_mode': mode, 'val_acc': val_acc, 'poison_acc': poison_acc, 'cum_poison_acc_mean': cum_poison}
                row.update(cuckoo)
                rows.append(row)
                current_round = None; val_acc = poison_acc = cum_poison = None
    return rows


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else 'output_logs'
    files = glob.glob(os.path.join(folder, '*console_output.txt'))
    all_rows = []
    for p in files:
        all_rows += parse_file(p)
    out = os.path.join(folder, 'CUCKOO_V1_METRICS_SUMMARY.csv')
    fields = ['file','round','attack_mode','val_acc','poison_acc','cum_poison_acc_mean','lambda_t','host_norm','poison_norm','egg_raw_norm','egg_selected_norm','cuckoo_norm','cosine_host_cuckoo','sign_agreement_host_cuckoo','kept_fraction','classifier_kept_fraction','target_row_kept_fraction','base_row_kept_fraction','memory_norm']
    with open(out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r.get(k, '') for k in fields})
    print(f'Wrote {len(all_rows)} rows to {out}')

if __name__ == '__main__':
    main()

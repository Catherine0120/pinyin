import json
from tqdm import tqdm
import time

reverse_table = {}

with open("./refactored/TriCntStat(dch-py-ch).txt", "r", encoding="gbk") as f:
    table = json.load(f)

for key, val in tqdm(table.items(), desc="processing entries", unit="entries"):
    cnt = 0
    small_table = {}
    for k, v in val.items():
        cnt += v
    for k, v in val.items():
        v = round(v / cnt, 6)
        small_table[k] = v
    reverse_table[key] = small_table

with open("./refactored/TriProbStat(dch-ch).txt", "w", encoding="gbk") as f:
    f.write(json.dumps(reverse_table, ensure_ascii=False, indent=4))
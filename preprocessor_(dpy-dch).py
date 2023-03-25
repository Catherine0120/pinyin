from pathlib import Path
import json
from tqdm import tqdm
from pypinyin import lazy_pinyin
from collections import defaultdict

dprob = defaultdict(lambda: defaultdict(float)) # "BiProbStat(dpy-dch).txt"

with open(Path.cwd()/"refactored"/"BiCntStat(ch-ch).txt", "r", encoding="gbk") as fin:
    file = json.load(fin)

with open(Path.cwd()/"refactored"/"BiProbStat(dpy-dch).txt", "w", encoding="gbk") as f:
    for first in tqdm(file, desc="generating 'BiProbStat(dpy-dch).txt'", unit="phrases"):
        for second, freq in file[first].items():
                dch = first + second
                pinyin_first = lazy_pinyin(first)[0]
                pinyin_second = lazy_pinyin(second)[0]
                dpy = pinyin_first + " " + pinyin_second
                dprob[dpy][" "] += freq
                dprob[dpy][dch] += freq
    for dpy in tqdm(dprob, desc="cnt => prob...", unit="entries"):
        for dch in dprob[dpy]:
            if dch != " ":
                dprob[dpy][dch] = round(dprob[dpy][dch] / dprob[dpy][" "], 6)
        del dprob[dpy][" "]
    
    for entry in tqdm(dprob, desc="sorting...", unit="entries"):
        s = dict(sorted(dprob[entry].items(), key=lambda x: x[1], reverse=True))
        dprob[entry] = s
        

    f.write(json.dumps(dprob, ensure_ascii=False, indent=4))
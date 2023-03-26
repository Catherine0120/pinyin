from pathlib import Path
import json
from tqdm import tqdm
from pypinyin import lazy_pinyin
from collections import defaultdict
import functools
import time
import re
from os import listdir, path
from pathlib import Path

def metric(fn):
    """
    Decorator function to measure the execution time of a function
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        print(f"[INFO ] Start executing {fn.__name__}()...")
        start_time = time.time()
        result = fn(*args, **kwargs)
        end_time = time.time()
        duration = round(end_time - start_time, 6) 
        print(f"        [Timer ] {fn.__name__}() finished in {duration} seconds")
        return result
    return wrapper


class Preprocessor(object):

    @metric
    def __init__(self, input_path):
        self.corpus = []
        self.dprob = defaultdict(lambda: defaultdict(float)) # "InitialBiProbStat(dpy-dch).txt"
        self.tprob = defaultdict(lambda: defaultdict(float)) # "InitialTriProbStat(tpy-tch).txt"
        
        if path.isdir(input_path):
            self.pathes = list(map(lambda x: path.join(input_path, x), listdir(input_path)))
        elif path.isfile(input_path):
            self.pathes = [input_path]
        else:
            raise ValueError("[Preprocessor.__init__()]: input_path is not a file or directory")

    @metric
    def parse_corpus(self):
        """ 
        parse corpus file 
        generate py-ch count tables
        """

        for i, file in enumerate(self.pathes):
            with open(file, "r", encoding="gbk", errors="ignore") as f:
                lines = f.readlines()
                for line in tqdm(lines, desc=f"parsing corpus[{i + 1}]... ", unit="lines"):
                    try:
                        data = json.loads(line.strip())
                        self.corpus.clear() # free memory
                        self.corpus += re.findall(r'[\u4e00-\u9fa5]+', data['html'].replace('原标题', ''))
                        self.corpus += re.findall(r'[\u4e00-\u9fa5]+', data['title'].replace(' ', ''))
                    except json.JSONDecodeError:
                        pass
                    self.parse()

    def parse(self):
        for line in self.corpus:
            if len(line) < 2:
                continue
            else:
                dch = line[0] + line[1]
                dpy = (lazy_pinyin(line[0]))[0] + " " + (lazy_pinyin(line[1]))[0]
                self.dprob[dpy][" "] += 1
                self.dprob[dpy][dch] += 1

            if len(line) < 3:
                continue
            else:
                tch = line[0] + line[1] + line[2]
                tpy = lazy_pinyin(line[0])[0] + " " + lazy_pinyin(line[1])[0] + " " + lazy_pinyin(line[2])[0]
                self.tprob[tpy][" "] += 1
                self.tprob[tpy][tch] += 1

    @metric
    def generate_files(self):
        for dpy in tqdm(self.dprob, desc="[dprob]: cnt => prob ", unit="entries"):
            for dch in self.dprob[dpy]:
                if dch != " ":
                    self.dprob[dpy][dch] = round(self.dprob[dpy][dch] / self.dprob[dpy][" "], 6)
            del self.dprob[dpy][" "]
        for entry in tqdm(self.dprob, desc="[dprob]: sorting... ", unit="entries"):
            s = dict(sorted(self.dprob[entry].items(), key=lambda x: x[1], reverse=True))
            self.dprob[entry] = s

        for tpy in tqdm(self.tprob, desc="[tprob]: cnt => prob ", unit="entries"):
            for tch in self.tprob[tpy]:
                if tch != " ":
                    self.tprob[tpy][tch] = round(self.tprob[tpy][tch] / self.tprob[tpy][" "], 6)
            del self.tprob[tpy][" "]
        for entry in tqdm(self.tprob, desc="[tprob]: sorting... ", unit="entries"):
            s = dict(sorted(self.tprob[entry].items(), key=lambda x: x[1], reverse=True))
            self.tprob[entry] = s

    def run(self):
        self.parse_corpus()
        self.generate_files()
        with open(Path.cwd()/"refactored"/"InitialBiProbStat(dpy-dch).txt", "w", encoding="gbk") as f:
            f.write(json.dumps(self.dprob, ensure_ascii=False, indent=4))    
        del self.dprob
        with open(Path.cwd()/"refactored"/"InitialTriProbStat(tpy-tch).txt", "w", encoding="gbk") as f:
            f.write(json.dumps(self.tprob, ensure_ascii=False, indent=4))    
        del self.tprob
    

if __name__ == "__main__":
    myPreprocessor = Preprocessor(Path.cwd()/"corpus")
    myPreprocessor.run()
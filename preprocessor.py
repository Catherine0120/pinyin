""" 
pre-process the corpus

Input: "./corpus/X.txt", X.txt records corpus drawn from sina
Output: "./refactored/XProbStat.txt", X indicating how many words

"""

import functools
import json
import time
import re
from tqdm import tqdm
from os import listdir, path
from pathlib import Path
from collections import defaultdict
from pypinyin import lazy_pinyin
from abc import ABC, abstractmethod

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


class Preprocessor(ABC):
    """
    Preprocess corpus data
    """

    @metric
    def __init__(self, input_path):
        self.corpus = []
        self.count = defaultdict(int)

        self.save_path = Path.cwd()/"refactored"/"SingleCntStat.txt"

        if path.isdir(input_path):
            self.pathes = list(map(lambda x: path.join(input_path, x), listdir(input_path)))
        elif path.isfile(input_path):
            self.pathes = [input_path]
        else:
            raise ValueError("[Preprocessor.__init__()]: input_path is not a file or directory")
        
        """ initialize "./refactored/SingleCntStat.txt" and "./data/pinyin_mapping.txt" """
        self.__process_table()
        self.__parse_corpus()
        self.__init_mapping()
        with open(self.save_path, "w", encoding="gbk") as f:
            f.write(json.dumps(self.count, ensure_ascii=False, indent=4))
        del self.count
    
    def __process_table(self):
        """ initialize "./data/pinyin_mapping.txt" using "look-up_table.txt" """
        address = Path.cwd()/"data"/"look-up_table.txt"
        destination = Path.cwd()/"data"/"pinyin_mapping.txt"
        table = {}
        with open(address, "r", encoding="gbk", errors="ignore") as f:
            for line in tqdm(f, desc="processing 'look-up_table.txt'", unit="lines"):
                key = (line.strip().split())[0]
                values = (line.strip().split())[1:]
                table[key] = {v: 0 for v in values}
        with open(destination, "w", encoding="gbk") as f:
            f.write(json.dumps(table, ensure_ascii=False, indent=4))

    def __parse_corpus(self):
        """ parse corpus file """
        for file in tqdm(self.pathes, desc="generating 'SingleCntStat.txt'", unit="files"):
            with open(file, "r", encoding="gbk", errors="ignore") as f:
                line = f.readline()
                while line:
                    try:
                        data = json.loads(line.strip())
                        self.corpus.clear() # free memory
                        self.corpus += re.findall(r'[\u4e00-\u9fa5]+', data['html'].replace('原标题', ''))
                        self.corpus += re.findall(r'[\u4e00-\u9fa5]+', data['title'].replace(' ', ''))
                    except json.JSONDecodeError:
                        pass
                    for newline in self.corpus:
                        for i in range(len(newline)):
                            self.count[newline[i]] += 1
                    line = f.readline()
    
    def __init_mapping(self):
        print("initializing 'pinyin_mapping.txt'")
        with open(Path.cwd()/"data"/"pinyin_mapping.txt", "r+", encoding="gbk") as f:
            self.map = json.load(f)
        for word, value in self.count.items():
            for key in self.map:
                if word in self.map[key]:
                    self.map[key][word] += value
        for key, my_dict in self.map.items():
            my_dict = dict(sorted(my_dict.items(), key=lambda x: x[1], reverse=True))
            self.map[key] = my_dict
        for key, my_dict in self.map.items():
            total = sum(my_dict.values())
            if total == 0:
                total = len(my_dict)
            for word in my_dict:
                my_dict[word] = round(my_dict[word] / total, 6)
        with open(Path.cwd()/"data"/"pinyin_mapping.txt", "w", encoding="gbk") as f:
            f.write(json.dumps(self.map, ensure_ascii=False, indent=4))
        del self.map

    @metric
    def parse_corpus(self):
        """ parse corpus file """
        for file in tqdm(self.pathes, desc="parsing corpus", unit="files"):
            with open(file, "r", encoding="gbk", errors="ignore") as f:
                line = f.readline()
                while line:
                    try:
                        data = json.loads(line.strip())
                        self.corpus.clear() # free memory
                        self.corpus += re.findall(r'[\u4e00-\u9fa5]+', data['html'].replace('原标题', ''))
                        self.corpus += re.findall(r'[\u4e00-\u9fa5]+', data['title'].replace(' ', ''))
                    except json.JSONDecodeError:
                        pass
                    self.parse()
                    line = f.readline()

    @abstractmethod
    def parse(self):
        raise NotImplementedError("[Preprocessor.parse()]: method not implemented")

    @abstractmethod
    def run(self):
        raise NotImplementedError("[Preprocessor.run()]: method not implemented")

class BiWordPreprocessor(Preprocessor):
    """ preprocessor based on binary grammar """

    def __init__(self, input_path):
        super().__init__(input_path)
        self.count = defaultdict(lambda: defaultdict(int)) # "BiCntStat(ch-ch).txt"
        self.freq = defaultdict(lambda: defaultdict(lambda: defaultdict(int))) # "BiCntStat(ch-py-ch)"
        self.prob = defaultdict(lambda: defaultdict(lambda: defaultdict(float))) # "BiProbStat(ch-py-ch)"
        self.dprob = defaultdict(lambda: defaultdict(float)) # "BiProbStat(dpy-dch).txt"
        self.save_path = Path.cwd()/"refactored"/"BiCntStat(ch-ch).txt"

    def parse(self):
        for line in self.corpus:
            for i in range(len(line) - 1):
                self.count[line[i]][line[i+1]] += 1
    
    @metric
    def calc_prob_chpych(self):
        self.save_path = Path.cwd()/"refactored"/"BiProbStat(ch-py-ch).txt"
        for first in tqdm(self.count, desc="generating 'BiProbStat(ch-py-ch).txt'", unit="phrases"):
            for second, freq in self.count[first].items():
                # TODO: 多音字处理
                # if len(lazy_pinyin(second)) != 1:
                #     print(lazy_pinyin(second))
                pinyin = lazy_pinyin(second)[0]
                self.freq[first][pinyin][" "] += freq
                self.freq[first][pinyin][second] += freq
        for first in self.freq:
            for second in self.freq[first]:
                for character in self.freq[first][second]:
                    self.prob[first][second][character] = round(self.freq[first][second][character] / self.freq[first][second][" "], 6)
                del self.prob[first][second][" "]
        del self.freq
        with open(self.save_path, "w", encoding="gbk") as f:
            f.write(json.dumps(self.prob, ensure_ascii=False, indent=4))    
        del self.prob  

    @metric
    def calc_prob_dpydch(self):
        self.save_path = Path.cwd()/"refactored"/"BiProbStat(dpy-dch).txt"
        for first in tqdm(self.count, desc="generating 'BiProbStat(dpy-dch).txt'", unit="phrases"):
            for second, freq in self.count[first].items():
                dch = first + second
                pinyin_first = lazy_pinyin(first)[0]
                pinyin_second = lazy_pinyin(second)[0]
                dpy = pinyin_first + " " + pinyin_second
                self.dprob[dpy][" "] += freq
                self.dprob[dpy][dch] += freq
        for dpy in tqdm(self.dprob, desc="cnt => prob...", unit="entries"):
            for dch in self.dprob[dpy]:
                if dch != " ":
                    self.dprob[dpy][dch] = round(self.dprob[dpy][dch] / self.dprob[dpy][" "], 6)
            del self.dprob[dpy][" "]
        for entry in tqdm(self.dprob, desc="sorting...", unit="entries"):
            s = dict(sorted(self.dprob[entry].items(), key=lambda x: x[1], reverse=True))
            self.dprob[entry] = s
    
    def run(self):
        self.parse_corpus()
        with open(self.save_path, "w", encoding="gbk") as f:
            f.write(json.dumps(self.count, ensure_ascii=False, indent=4))
        self.calc_prob_chpych()
        self.calc_prob_dpydch()
        del self.count

class TriWordPreprocessor(Preprocessor):
    """ preprocessor based on triple grammar """

    def __init__(self, input_path):
        super().__init__(input_path)
        self.count = defaultdict(lambda: defaultdict(int)) # "TriCntStat(dch-ch).txt"
        self.freq = defaultdict(lambda: defaultdict(lambda: defaultdict(int))) # "TriCntStat(dch-py-ch)"
        self.prob = defaultdict(lambda: defaultdict(lambda: defaultdict(float))) # "TriProbStat(dch-py-ch)"
        self.save_path = Path.cwd()/"refactored"/"TriCntStat(dch-py-ch).txt"
    
    def parse(self):
        for line in self.corpus:
            for i in range(len(line) - 2):
                self.count[line[i] + line[i+1]][line[i+2]] += 1

    def run(self):
        self.parse_corpus()
        self.IO()
        self.calc_prob_dchpych()
        del self.count

    @metric
    def IO(self):
        with open(self.save_path, "w", encoding="gbk") as f:
            f.write(json.dumps(self.count, ensure_ascii=False, indent=4))
    
    @metric
    def calc_prob_dchpych(self):
        self.save_path = Path.cwd()/"refactored"/"TriProbStat(dch-py-ch).txt"
        for first in tqdm(self.count, desc="generating 'TriProbStat(dch-py-ch).txt'", unit="phrases"):
            for second, freq in self.count[first].items():
                # TODO: 多音字处理
                # if len(lazy_pinyin(second)) != 1:
                #     print(lazy_pinyin(second))
                pinyin = lazy_pinyin(second)[0]
                self.freq[first][pinyin][" "] += freq
                self.freq[first][pinyin][second] += freq
        for first in self.freq:
            for second in self.freq[first]:
                for character in self.freq[first][second]:
                    self.prob[first][second][character] = round(self.freq[first][second][character] / self.freq[first][second][" "], 6)
                del self.prob[first][second][" "]
        del self.freq
        with open(self.save_path, "w", encoding="gbk") as f:
            f.write(json.dumps(self.prob, ensure_ascii=False, indent=4))    
        del self.prob  

if __name__ == "__main__":
    # myPreprocessor = BiWordPreprocessor(Path.cwd()/"corpus")
    myPreprocessor = TriWordPreprocessor(Path.cwd()/"corpus")
    myPreprocessor.run()
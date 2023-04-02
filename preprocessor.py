""" 
pre-process the corpus

Input: "./corpus/X.txt", X.txt records corpus drawn from sina
Output: 
    0.  一元模型依赖文件（辅助文件）
        "./data/pinyin_mapping.txt": {"qing": {"清": 0.9, "情": 0.1}}
        "./refactored/SingleCntStat.txt": {"清": 900, "情": 100}
    1.  二元模型依赖文件
        "./refactored/BiCntStat(ch-ch).txt": {"清": {"华": 900, "划": 1000}}
        "./refactored/BiProbStat(ch-py-ch).txt": {"清": {"hua": {"华": 0.9, "划": 0.1}}}
        "./refactored/BiProbStat(dpy-dch).txt": {"qing hua": {"清华": 0.9, "情话": 0.1}}
        "./refactored/InitialBiProbStat(dpy-dch).txt": 同上, 但只统计每一个phrase的首字词
    2.  三元模型依赖文件（也包括部分以上二元模型依赖文件）
        "./refactored/TriCntStat(dch-py-ch).txt": {"清华": {"da": {"大": 900, "达": 100}}}
        "./refactored/TriProbStat(dch-py-ch).txt": {"清华": {"da": {"大": 0.9, "达": 0.1}}}
        "./refactored/InitialTriProbStat(tpy-tch).txt": 同上, 但只统计每一个phrase的首字词
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
        self.dprob = defaultdict(lambda: defaultdict(float)) # "InitialBiProbStat(dpy-dch).txt"
        self.tprob = defaultdict(lambda: defaultdict(float)) # "InitialTriProbStat(tpy-tch).txt"
        self.save_path = Path.cwd()/"refactored"/"SingleCntStat.txt"

        if path.isdir(input_path):
            self.pathes = list(map(lambda x: path.join(input_path, x), listdir(input_path)))
        elif path.isfile(input_path):
            self.pathes = [input_path]
        else:
            raise ValueError("[Preprocessor.__init__()]: input_path is not a file or directory")
        
        """ initialize "./refactored/SingleCntStat.txt" and "./data/pinyin_mapping.txt" """
        self.__process_table()
        self.__parse_corpus() # SingleCntStat save to self.count temporarily
        self.__init_mapping() # generate "pinyin_mapping.txt" and "SingleCntStat.txt"
        with open(self.save_path, "w", encoding="gbk") as f:
            f.write(json.dumps(self.count, ensure_ascii=False, indent=4))
        del self.count

        """ initialize "./refactored/InitialXProbStat.txt" """
        self.parse_corpus() # generate "InitialXProbStat.txt"
        self.generate_files()
    
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
        """ 
        parse corpus file 
        generate py-ch count tables
        """

        for i, file in enumerate(self.pathes):
            with open(file, "r", encoding="gbk", errors="ignore") as f:
                lines = f.readlines()
                for line in tqdm(lines, desc=f"parsing corpus[{i+1}]... ", unit="lines"):
                    try:
                        data = json.loads(line.strip())
                        self.corpus.clear() # free memory
                        self.corpus += re.findall(r'[\u4e00-\u9fa5]+', data['html'].replace('原标题', ''))
                        self.corpus += re.findall(r'[\u4e00-\u9fa5]+', data['title'].replace(' ', ''))
                    except json.JSONDecodeError:
                        pass
                    self.parse()

    @abstractmethod
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
        raise NotImplementedError("[Preprocessor.parse()]: method not implemented")

    @abstractmethod
    def run(self):
        raise NotImplementedError("[Preprocessor.run()]: method not implemented")

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
        with open(Path.cwd()/"refactored"/"InitialBiProbStat(dpy-dch).txt", "w", encoding="gbk") as f:
            f.write(json.dumps(self.dprob, ensure_ascii=False, indent=4))    
        del self.dprob

        for tpy in tqdm(self.tprob, desc="[tprob]: cnt => prob ", unit="entries"):
            for tch in self.tprob[tpy]:
                if tch != " ":
                    self.tprob[tpy][tch] = round(self.tprob[tpy][tch] / self.tprob[tpy][" "], 6)
            del self.tprob[tpy][" "]
        for entry in tqdm(self.tprob, desc="[tprob]: sorting... ", unit="entries"):
            s = dict(sorted(self.tprob[entry].items(), key=lambda x: x[1], reverse=True))
            self.tprob[entry] = s
        with open(Path.cwd()/"refactored"/"InitialTriProbStat(tpy-tch).txt", "w", encoding="gbk") as f:
            f.write(json.dumps(self.tprob, ensure_ascii=False, indent=4))    
        del self.tprob

    def calc_prob_chch(cntPath, probPath):
        reverse_table = {}
        with open(cntPath, "r", encoding="gbk") as f:
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

        with open(probPath, "w", encoding="gbk") as f:
            f.write(json.dumps(reverse_table, ensure_ascii=False, indent=4))
            

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
        with open(self.save_path, "w", encoding="gbk") as f:
            f.write(json.dumps(self.dprob, ensure_ascii=False, indent=4))    
        del self.dprob
    
    def run(self):
        self.parse_corpus()
        with open(self.save_path, "w", encoding="gbk") as f:
            f.write(json.dumps(self.count, ensure_ascii=False, indent=4))
        self.calc_prob_chpych()
        self.calc_prob_dpydch()
        self.calc_prob_chch("./refactored/BiCntStat(ch-ch).txt", "./refactored/BiProbStat(ch-ch).txt")
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
        self.calc_prob_chch("./refactored/TriCntStat(dch-py-ch).txt", "./refactored/TriProbStat(dch-ch).txt")
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
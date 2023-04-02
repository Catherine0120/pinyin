"""
generator HMM (Hidden Markov Model) graph for each input

Input: sentence written in pinyin
Output: HMM graph and viterbi method to calculate the most likely Chinese sentence

"""

import json
import math
from tqdm import tqdm
import time
from pathlib import Path
from typing import List
import functools

ALTERNATIVE = 50
BIG_NUMBER = 100
ALPHA = 0.75

def sigmoid(n: float) -> float:
    if n == 0:
        return -math.inf
    else:
        return math.log(n)
    
def sortTangleList(a: List[List], b: List[float]):
    pairs = [(a, b) for a, b in zip(a, b)]
    pairs.sort(key=lambda x: x[1], reverse=True)
    a = [pair[0] for pair in pairs[0:min(ALTERNATIVE, len(pairs))]]
    b = [pair[1] for pair in pairs[0:min(ALTERNATIVE, len(pairs))]]
    return (a, b)

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

class Graph(object):
    """ character graph """

    def __init__(self, output):
        self.output_path = output
        self._IO()
        self.sentence = " "

    @metric
    def _IO(self):
        with open(Path.cwd()/"pinyin_mapping.txt", "r", encoding="gbk") as f:
            self.mapping = json.load(f)
        with open(Path.cwd()/"refactored"/"BiProbStat(ch-ch).txt", "r", encoding="gbk") as f:
            self.nstat = json.load(f)
        with open("output.txt", "w", encoding="gbk") as f:
            pass
        with open(Path.cwd()/"refactored"/"InitialBiProbStat(dpy-dch).txt", "r", encoding="gbk") as bf:
            self.bi_init = json.load(bf)

    def init_sentence(self, paths, scores):
        paths = [] # record the possible path: List
        scores = [] # record the corresponding score for each path: float
        self.init_flag = 0

        bi_pinyin = self.sentence[0] + " " + self.sentence[1]
        try:
            bi_dict = self.bi_init[bi_pinyin]
        except:
            self.init_flag = 1

        if self.init_flag == 0:
            for bi_phrase, bi_val in bi_dict.items():
                try:
                    single_val1 = self.mapping[self.sentence[0]][bi_phrase[0]]
                    single_val2 = self.mapping[self.sentence[1]][bi_phrase[1]]
                except:
                    continue
                score = (1 - ALPHA) * sigmoid(bi_val) + ALPHA * sigmoid((single_val1 * single_val2) ** (1/2))
                path = [bi_phrase[0], bi_phrase[1]]
                paths.append(path)
                scores.append(score)
        else:
            single_dict = self.mapping[self.sentence[0]]
            tmp = 0
            for word, value in single_dict.items():
                tmp += 1
                path = [word]
                paths.append(path)
                scores.append(value)
                if tmp == ALTERNATIVE:
                    break
            paths = paths.copy()
            scores = scores.copy()
        
        tp = sortTangleList(paths, scores)
        paths = tp[0]
        scores = tp[1]
        try:
            del bi_dict
        except:
            pass

        return (paths, scores)
    
    def _viterbi(self):
        """
        find the max likelihood solution to pinyin translation
        """

        paths = [] # record the possible path: List
        scores = [] # record the corresponding score for each path: float

        """ initialize the first characters of a sentence """
        paths, scores = self.init_sentence(paths, scores)

        for i in range(2 - self.init_flag, self.n_layer):
            cur_paths = []
            cur_scores = []
            for index, path in enumerate(paths):
                new_paths = []
                new_scores = []
                for character, prob in self.mapping[self.sentence[i]].items():
                    try:
                        score = (1 - ALPHA) * sigmoid(prob) + ALPHA * sigmoid(self.nstat[path[-1]][character])
                        new_paths.append(path + [character])
                        new_scores.append(scores[index] + score)
                    except:
                        continue
                tp = sortTangleList(new_paths, new_scores)
                cur_paths += tp[0]
                cur_scores += tp[1]

            if (len(cur_paths) == 0):
                for index, path in enumerate(paths):
                    new_paths = []
                    new_scores = []
                    for character, prob in self.mapping[self.sentence[i]].items():
                        score = (1 - ALPHA) * sigmoid(prob) - ALPHA * BIG_NUMBER
                        new_paths.append(path + [character])
                        new_scores.append(scores[index] + score)
                    tp = sortTangleList(new_paths, new_scores)
                    cur_paths += tp[0]
                    cur_scores += tp[1]

            tp = sortTangleList(cur_paths, cur_scores)
            paths = tp[0]
            scores = tp[1]

        with open(self.output_path, "a", encoding="gbk") as f:
            try:
                f.write("".join(paths[0]))
                f.write("\n")
            except:
                f.write("\n")

    
    def run(self, sentence: List[str]):
        self.sentence = sentence
        self.n_layer = len(sentence)
        self._viterbi()


def bi_gram_generator_2(input_path: str, output: str):
    graph = Graph(output)
    start_time = time.time()
    with open(Path.cwd()/input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in tqdm(lines, desc="processing each sentence... ", unit="lines"):
            sentence = line.strip().split(" ")
            graph.run(sentence)
    end_time = time.time()
    duration = round(end_time - start_time, 6) 
    print(f"[Timer ] program finished in {duration} seconds")
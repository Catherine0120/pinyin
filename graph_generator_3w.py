"""
generator HMM (Hidden Markov Model) graph for each input
using Triple Model method

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
from collections import defaultdict

ALTERNATIVE = 20 # how many nodes in each layer are considered
PARAMETER = [0.75, 0.2, 0.05] # weights for three conditional probabilities 
PARAMETER2 = [0.4, 0.6]

""" 
[log]: (子准确率，词准确率)
PARAMETER = [0.75, 0.2, 0.05]
PARAMETER2 = [0.4, 0.6]
    ALTERNATIVE = 10: (0.8793, 0.5070)
    ALTERNATIVE = 15: (0.8795, 0.5050)
    ALTERNATIVE = 18: (0.8797, 0.5070)
    ALTERNATIVE = 20: (0.8799, 0.5090) ✓
    ALTERNATIVE = 25: (0.8795, 0.5090)
    ALTERNATIVE = 30: (0.8786, 0.5090)

ALTERNATIVE = 25
PARAMETER2 = [0.4, 0.6]
    PARAMETER = [0.2, 0.6, 0.2]: (0.8701, 0.4431)
    PARAMETER = [0.3, 0.4, 0.3]: (0.8704, 0.4611)
    PARAMETER = [0.3, 0.5, 0.2]: (0.8749, 0.4671)
    PARAMETER = [0.4, 0.4, 0.1]: (0.8776, 0.4850) 
    PARAMETER = [0.5, 0.3, 0.2]: (0.8772, 0.4890) 
    PARAMETER = [0.6, 0.3, 0.1]: (0.8788, 0.4950) 
    PARAMETER = [0.7, 0.2, 0.1]: (0.8793, 0.5050) 
    PARAMETER = [0.75, 0.2, 0.05]: (0.8795, 0.5090) ✓
    PARAMETER = [0.75, 0.1, 0.15]: (0.8776, 0.5030)
    PARAMETER = [0.8, 0.15, 0.05]: (0.8790, 0.5110) 
    PARAMETER = [0.9, 0.08, 0.02]: (0.8779, 0.5070)

ALTERNATIVE = 25
PARAMETER = [0.3, 0.5, 0.2]
    PARAMETER2 = [0.9, 0.1]: (0.8727, 0.4631)
    PARAMETER2 = [0.8, 0.2]: (0.8739, 0.4651)
    PARAMETER2 = [0.5, 0.5]: (0.8743, 0.4651)
    PARAMETER2 = [0.4, 0.6]: (0.8749, 0.4671) ✓
    PARAMETER2 = [0.3, 0.7]: (0.8744, 0.4671)
    PARAMETER2 = [0.2, 0.8]: (0.8741, 0.4671)
    PARAMETER2 = [0.1, 0.9]: (0.8738, 0.4671)
"""

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
        with open(Path.cwd()/"data"/"pinyin_mapping.txt", "r", encoding="gbk") as f:
            self.mapping = json.load(f)
        with open(Path.cwd()/"refactored"/"BiProbStat(ch-py-ch).txt", "r", encoding="gbk") as f:
            self.dstat = json.load(f)
        with open(Path.cwd()/"refactored"/"TriProbStat(dch-py-ch).txt", "r", encoding="gbk") as f:
            self.tstat = json.load(f)
        with open(self.output_path, "w", encoding="gbk") as f:
            pass
        with open(Path.cwd()/"refactored"/"InitialBiProbStat(dpy-dch).txt", "r", encoding="gbk") as bf:
            self.bi_init = json.load(bf)
        with open(Path.cwd()/"refactored"/"InitialTriProbStat(tpy-tch).txt", "r", encoding="gbk") as tf:
            self.tri_init = json.load(tf)

    def _viterbi(self):
        paths = []
        scores = []
        flag = 0
        init_flag = True
        bi_dict2_flag = True

        if self.n_layer < 3:
            print("[catch ERROR]: sentence length smaller than 3")
            with open(self.output_path, "a", encoding="gbk") as f:
                f.write("\n")
            return

        """ initialize the first characters of a sentence """
        bi_pinyin1 = self.sentence[0] + " " + self.sentence[1]
        bi_pinyin2 = self.sentence[1] + " " + self.sentence[2]
        tri_pinyin = self.sentence[0] + " " + self.sentence[1] + " " + self.sentence[2]
        try:
            bi_dict1 = self.bi_init[bi_pinyin1]
        except:
            init_flag = False

        if init_flag == True: # can find the possible dch for the initial dpy
            try:
                bi_dict2 = self.bi_init[bi_pinyin2]
            except:
                bi_dict2_flag = False
            try:
                tri_dict = self.tri_init[tri_pinyin]    
                for tri_phrase, tri_val in tri_dict.items():
                    try:
                        bi_val1 = bi_dict1[tri_phrase[0:2]]
                        single_val1 = self.mapping[self.sentence[0]][tri_phrase[0]]
                        single_val2 = self.mapping[self.sentence[1]][tri_phrase[1]]
                        single_val3 = self.mapping[self.sentence[2]][tri_phrase[2]]
                    except:
                        continue
                    if bi_dict2_flag == True:
                        try:
                            bi_val2 = bi_dict2[tri_phrase[1:3]]
                        except:
                            bi_val2 = bi_val1
                    else:
                        bi_val2 = bi_val1

                    score = PARAMETER[0] * sigmoid(tri_val) + \
                            PARAMETER[1] * sigmoid((bi_val1 * bi_val2) ** (1/2)) + \
                            PARAMETER[2] * sigmoid((single_val1 * single_val2 * single_val3) ** (1/3))
                    path = [tri_phrase[0], tri_phrase[1], tri_phrase[2]]
                    paths.append(path)
                    scores.append(score)
            except:
                pass

            if len(paths) == 0: # cannot generate tri_dict
                flag = 1
                for bi_phrase, bi_val in bi_dict1.items():
                    try:
                        single_val1 = self.mapping[self.sentence[0]][bi_phrase[0]]
                        single_val2 = self.mapping[self.sentence[1]][bi_phrase[1]]
                    except:
                        continue
                    score = PARAMETER2[0] * sigmoid(bi_val) + \
                            PARAMETER2[1] * sigmoid((single_val1 * single_val2) ** (1/2))
                    path = [bi_phrase[0], bi_phrase[1]]
                    paths.append(path)
                    scores.append(score)

        else: # handle exception when first dch cannot be determined
            flag = 1
            single_dict = self.mapping[self.sentence[0]]
            tmp = 0
            for word, value in single_dict.items():
                tmp += 1
                path = [word]
                paths.append(path)
                scores.append(value)
                if tmp == ALTERNATIVE:
                    break
            cur_paths = []
            cur_scores = []
            for index, path in enumerate(paths):
                try:
                    bi_dict = self.dstat[path[-1]][self.sentence[1]]
                except:
                    continue
                new_paths = []
                new_scores = []
                for word, val in bi_dict.items():
                    try:
                        single_prob = self.mapping[self.sentence[1]][word]
                    except:
                        continue
                    score = PARAMETER2[0] * sigmoid(val) + \
                            PARAMETER2[1] * sigmoid(single_prob)
                    new_paths.append(path + [word])
                    new_scores.append(scores[index] + score)
                cur_paths += new_paths
                cur_scores += new_scores
            paths = cur_paths.copy()
            scores = cur_scores.copy()

        # store results and free up memory
        tp = sortTangleList(paths, scores)
        paths = tp[0]
        scores = tp[1]
        try:
            del tri_dict
            del bi_dict1
            del bi_dict2
        except:
            pass

        """ find out the rest characters in the sentence """
        for i in range(3 - flag, self.n_layer):
            cur_paths = []
            cur_scores = []
            for index, path in enumerate(paths):
                try:
                    tri_dict = self.tstat[path[-2] + path[-1]][self.sentence[i]]
                except:
                    continue
                try:
                    bi_dict = self.dstat[path[-1]][self.sentence[i]]
                except:
                    continue
                
                new_paths = []
                new_scores = []
                for word, val in tri_dict.items():
                    try:
                        single_prob = self.mapping[self.sentence[i]][word]
                    except:
                        continue
                    score = PARAMETER[0] * sigmoid(val) + \
                            PARAMETER[1] * sigmoid(bi_dict[word]) + \
                            PARAMETER[2] * sigmoid(single_prob)
                    new_paths.append(path + [word])
                    new_scores.append(scores[index] + score)
                
                tp = sortTangleList(new_paths, new_scores)
                cur_paths += tp[0]
                cur_scores += tp[1]

            if len(cur_paths) == 0: # bi-model
                for index, path in enumerate(paths):
                    try:
                        bi_dict = self.dstat[path[-1]][self.sentence[i]]
                    except:
                        continue
                    new_paths = []
                    new_scores = []
                    for word, val in bi_dict.items():
                        try:
                            single_prob = self.mapping[self.sentence[i]][word]
                        except:
                            continue
                        score = PARAMETER2[0] * sigmoid(val) + \
                                PARAMETER2[1] * sigmoid(single_prob)
                        new_paths.append(path + [word])
                        new_scores.append(scores[index] + score)
                    
                    tp = sortTangleList(new_paths, new_scores)
                    cur_paths += tp[0]
                    cur_scores += tp[1]
            
            if len(cur_paths) == 0: # tri-model
                for index, path in enumerate(paths):
                    new_paths = []
                    new_scores = []
                    cnt = 0
                    for word, val in self.mapping[self.sentence[i]].items():
                        cnt += 1
                        new_paths.append(path + [word])
                        new_scores.append(scores[index] + sigmoid(val))
                        if cnt == ALTERNATIVE:
                            break
                    
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


def tri_gram_generator(input_path: str, output: str):
    graph = Graph(output)
    start_time = time.time()
    with open(Path.cwd()/input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in tqdm(lines, desc="processing each sentence... ", unit="lines"):
            sentence = line.strip().split(" ")
            graph.run(sentence)
    end_time = time.time()
    duration = round(end_time - start_time, 6) 
    print(f"[INFO ] program finished in {duration} seconds")
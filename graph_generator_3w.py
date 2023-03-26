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

ALTERNATIVE = 35 # how many nodes in each layer are considered

def sigmoid(n: float) -> float:
    if n == 0:
        return -math.inf
    else:
        return math.log(n)

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


class Node:
    """ the wrapped character to be selected in each layer """

    def __init__(self, pinyin, character, prob=0):
        self.pinyin = pinyin
        self.character = character
        self.prob = prob # [0, 1]


class Graph(object):
    """ character graph """

    def __init__(self):
        self._IO()
        self.sentence = " "
    
    @metric
    def _IO(self):
        with open(Path.cwd()/"data"/"pinyin_mapping.txt", "r", encoding="gbk") as f:
            self.mapping = json.load(f)
        with open(Path.cwd()/"reafactored"/"BiProbStat(ch-py-ch).txt", "r", encoding="gbk") as f:
            self.dstat = json.load(f)
        with open(Path.cwd()/"refactored"/"TriProbStat(dch-py-ch).txt", "r", encoding="gbk") as f:
            self.tstat = json.load(f)
        with open("output.txt", "w", encoding="gbk") as f:
            pass
    
    def _generate_graph(self, sentence: List[str]):
        """
        generate the graph
        self.graph[i][j] means the jth node in the ith layer
        """

        self.sentence = sentence
        self.graph = []
        self.debug_graph = []
        self.n_layer = len(sentence)

        for pinyin in sentence:
            layer = []
            debug_layer = []
            for i in range(0, ALTERNATIVE):
                try:
                    character = list(self.mapping[pinyin].keys())[i]
                    layer.append(Node(pinyin, character, self.mapping[pinyin][character]))
                    debug_layer.append(character)
                except: 
                    character = " "
                    layer.append(Node(pinyin, character))
                    debug_layer.append(character)
            self.graph.append(layer)
            self.debug_graph.append(debug_layer)
        
        print(self.debug_graph)

    def _viterbi(self):
        pass
    
    def run(self, sentence: List[str]):
        self._generate_graph(sentence)
        self._viterbi()
        del self.graph
        pass


if __name__ == "__main__":
    graph = Graph()
    start_time = time.time()
    with open(Path.cwd()/"input.txt", "r", encoding="utf-8") as f:
        line = f.readline()
        while line:
            sentence = line.strip().split(" ")
            graph.run(sentence)
            line = f.readline()
    end_time = time.time()
    duration = round(end_time - start_time, 6) 
    print(f"[INFO ] program finished in {duration} seconds")
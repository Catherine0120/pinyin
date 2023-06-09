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

ALTERNATIVE = 10
RANK = 5

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
    """ the character selected in each layer """

    def __init__(self, pinyin, character, prob=-math.inf):
        self.pinyin = pinyin
        self.character = character
        self.prob = prob # maximum probability from the start node
        self.next = [] # tuples (index of node in the next layer, conditional probability)

class Graph(object):
    """ character graph """

    def __init__(self):
        self._IO()
        self.sentence = " "

    @metric
    def _IO(self):
        with open(Path.cwd()/"pinyin_mapping.txt", "r", encoding="gbk") as f:
            self.mapping = json.load(f)
        with open(Path.cwd()/"refactored"/"BiProbStat(ch-py-ch).txt", "r", encoding="gbk") as f:
            self.stat = json.load(f)

    # @metric   
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
    
    # @metric
    def _generate_transition_matrix(self):
        """ 
        generate the transition matrix
        self.trans[i][j][k] means the conditional probability of jth node in the ith layer and kth node in the (i+1)th layar
        """

        self.trans = [] # list (with len(self.graph) entries of 2-D lists 
        for i in range(len(self.graph) - 1):
            prob_table = [[0 for j in range(ALTERNATIVE)] for k in range(ALTERNATIVE)]
            for j in range(0, ALTERNATIVE):
                for k in range(0, ALTERNATIVE):
                    try:
                        prob_table[j][k] = self.stat[self.graph[i][j].character][self.graph[i+1][k].pinyin][self.graph[i+1][k].character]
                    except:
                        prob_table[j][k] = 0
            self.trans.append(prob_table)   
    
    def _viterbi(self, alpha=0.5) -> list: # TODO
        """
        find the max likelihood solution to pinyin translation
        """

        for layer in range(len(self.trans)):
            max_prob = 0
            for j in range(len(self.trans[layer])):
                for k in range(len(self.trans[layer][j])):
                    cur = self.trans[layer][j][k]
                    if cur >= (max_prob / 8) and cur > 0:
                        while (len(self.graph[layer][j].next) > 0) and (self.graph[layer][j].next[-1][1] < (cur / 5)):
                            self.graph[layer][j].next.pop()
                        self.graph[layer][j].next.append((k, cur))
                        max_prob = cur
                self.graph[layer][j].next = sorted(self.graph[layer][j].next, key=lambda x: x[1], reverse=True)

        paths = [] # record the possible path: List
        score = [] # record the corresponding score for each path: float
        debug_paths = []

        for i in range(len(self.graph) - 1):
            if i == 0:
                group = [] # tuple(i, (j, t)): the test score between i in layer and j in next-layer
                for j in range(0, ALTERNATIVE):
                    node = self.graph[i][j]
                    group += [(j, (node.next[x][0], round(alpha * node.next[x][1] + (1 - alpha) * node.prob, 6))) for x in range(len(node.next))]
                group = sorted(group, key=lambda x: x[1][1], reverse=True)
                group = group[0:min(RANK, len(group))]
                for j in range(0, min(RANK, len(group))):
                    paths.append([group[j][0], group[j][1][0]])
                    score.append(group[j][1][1])
                    debug_paths.append([self.graph[i][group[j][0]].character, self.graph[i+1][group[j][1][0]].character])
            else:
                group = [] # tuple(path, t): the test score of path
                for j in range(0, len(paths)):
                    path = paths[j]
                    for k in self.graph[i][path[i]].next:
                        new_path = path.copy()
                        new_path.append(k[0])
                        t = round(score[j] * k[1], 6)
                        group.append((new_path, t))
                if len(group) < 1:
                    for i in range(0, min(len(paths), 3)):
                        for j in range(0, 2):
                            new_path = paths[i].copy()
                            new_path.append(j)
                            t = round(score[0] * self.graph[i+1][j].prob, 6)
                            group.append((new_path, t))
                    
                group = sorted(group, key=lambda x: x[1], reverse=True)
                group = group[0:min(RANK, len(group))]
                paths.clear()
                score.clear()
                debug_paths.clear()
                for j in range(0, min(RANK, len(group))):
                    paths.append(group[j][0])
                    score.append(group[j][1])
                    debug_paths.append([self.graph[key][group[j][0][key]].character for key in range(0, len(group[j][0]))])
        return debug_paths[0]

    
    def _debug_print_graph(self):
        for i in range(len(self.graph)):
            print(f"layer[{i}]: {self.sentence[i]}")
            for j in range(0, ALTERNATIVE - 1):
                node = self.graph[i][j]
                if len(node.next) != 0:
                    print(f"    {node.character}: ", end=" ")
                    for x in node.next:
                        print(f"{self.graph[i+1][x[0]].character}({x[1]})", end=" ")
                    print("\n", end="")
        print()
    
    def run(self, sentence: List[str]) -> list:
        self._generate_graph(sentence)
        self._generate_transition_matrix()
        return self._viterbi()


def bi_gram_generator(input_path: str, output: str):
    graph = Graph()
    start_time = time.time()
    with open(Path.cwd()/input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in tqdm(lines, desc="processing each sentence... ", unit="lines"):
            sentence = line.strip().split(" ")
            ans = graph.run(sentence)
            with open(output, "a", encoding="gbk") as fl:
                fl.write("".join(ans))
                fl.write("\n")
    end_time = time.time()
    duration = round(end_time - start_time, 6) 
    print(f"[Timer ] program finished in {duration} seconds")
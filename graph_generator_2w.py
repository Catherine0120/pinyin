"""
generator HMM (Hidden Markov Model) graph for each input
using Binary Model method

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
RANK = 25 # phrases ranking the first RANKth are considered
JUDGE = 2 # the 1st to JUDGEth largest in "BiProbStat(dpy-dch)"
BAR = [0.3, 0.2] # should be greater than (BARi)th

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
        self.next = [] # tuples (index of node in the next layer, conditional probability)

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
            self.stat = json.load(f)
        with open(Path.cwd()/"refactored"/"BiProbStat(dpy-dch).txt", "r", encoding="gbk") as f:
            self.check = json.load(f)
        with open(self.output_path, "w", encoding="gbk") as f:
            pass

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
        self.sentence_check = defaultdict(lambda x: defaultdict(float))
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
        for i in range(len(sentence) - 1):
            phrase = sentence[i] + " " + sentence[i+1]
            try:
                self.sentence_check[phrase] = self.check[phrase]
            except:
                continue

        # print(self.debug_graph)
    
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
    
    # @metric
    def _viterbi(self, alpha=0.9, beta=0.2): # TODO
        """
        find the max likelihood solution to pinyin translation
        """

        for layer in range(len(self.trans)):
            max_prob = 0
            for j in range(len(self.trans[layer])):
                for k in range(len(self.trans[layer][j])):
                    cur = self.trans[layer][j][k]
                    if cur >= (max_prob / 8) and cur > 0:
                        while (len(self.graph[layer][j].next) > 0) and (self.graph[layer][j].next[-1][1] < (cur / 4)):
                            self.graph[layer][j].next.pop()
                        self.graph[layer][j].next.append((k, cur))
                        max_prob = cur
                self.graph[layer][j].next = sorted(self.graph[layer][j].next, key=lambda x: x[1], reverse=True)

        paths = [] # record the possible path: List
        score = [] # record the corresponding score for each path: float
        debug_paths = []

        flag = True # whether jieba or not, flag == True indicates that it probability encounters a jieba

        for i in range(len(self.graph) - 1):
            if i != 0 and flag == True:
                group = [] # tuple(k, j, (p, t)): the test score among kth path in previous layers, j in this layer, and p in next layer 
                for j in range(0, ALTERNATIVE):
                    node = self.graph[i][j]
                    for k in range(len(paths)): # TODO
                        path = paths[k]
                        group += [(k, j, (node.next[x][0], round( \
                                                        alpha * sigmoid(node.next[x][1]) \
                                                        + (1 - alpha) * (beta * sigmoid(node.prob) \
                                                        + (1 - beta) * sigmoid(self.trans[i-1][path[-1]][j])), 6))) for x in range(len(node.next))]
                    if len(group) == 0:
                        group += [(0, j, (0, round(score[0], 6)))]
                group = sorted(group, key=lambda x: x[2][1], reverse=True)
                group = group[0:min(RANK, len(group))]
                new_paths = []
                new_score = []
                debug_paths.clear()
                for minor_path in group:
                    new_path = paths[minor_path[0]].copy()
                    new_path.append(minor_path[1])
                    new_path.append(minor_path[2][0])
                    debug_paths.append([self.graph[key][new_path[key]].character for key in range(0, len(new_path))])
                    new_paths.append(new_path)
                    new_score.append(round(score[minor_path[0]] + minor_path[2][1], 6))
                score = new_score
                paths.clear()
                paths = new_paths
                del new_paths
                flag = False
                continue
            
            flag = True

            try:
                dict = self.sentence_check[self.sentence[i] + " " + self.sentence[i + 1]]
                for j, key in enumerate(dict):
                    if j == JUDGE:
                        break
                    bar_value = dict[key]
                    if bar_value > BAR[j]: # no jieba
                        flag = False
            except:
                flag = False
                pass

            if i == 0:
                group = [] # tuple(i, (j, t)): the test score between i in layer and j in next-layer
                for j in range(0, ALTERNATIVE):
                    node = self.graph[i][j]
                    group += [(j, (node.next[x][0], round(alpha * sigmoid(node.next[x][1]) + (1 - alpha) * sigmoid(node.prob), 6))) for x in range(len(node.next))]
                group = sorted(group, key=lambda x: x[1][1], reverse=True)
                group = group[0:min(RANK, len(group))]
                for j in range(0, min(RANK, len(group))):
                    paths.append([group[j][0], group[j][1][0]])
                    score.append(group[j][1][1])
                    debug_paths.append([self.graph[i][group[j][0]].character, self.graph[i+1][group[j][1][0]].character])
                flag = False
            elif flag == True and i != len(self.graph) - 2 and i != len(self.graph) - 1:
                continue
            else:
                group = [] # tuple(path, t): the test score of path
                for j in range(0, len(paths)):
                    path = paths[j]
                    if len(self.graph[i][path[i]].next) == 0:
                        for k in range(len(self.graph[i+1])):
                            new_path = path.copy()
                            new_path.append(k)
                            t = round(score[j] + alpha * sigmoid(self.graph[i][j].prob) + (1 - alpha) * sigmoid(self.trans[i][path[-1]][k]), 6)
                            group.append((new_path, t))
                    else:
                        for k in self.graph[i][path[i]].next:
                            new_path = path.copy()
                            new_path.append(k[0])
                            t = round(score[j] + alpha * sigmoid(k[1]) + (1 - alpha) * sigmoid(self.graph[i][path[i]].prob), 6)
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


        with open(self.output_path, "a", encoding="gbk") as f:
            f.write("".join(debug_paths[0]))
            if i != self.n_layer - 1:
                f.write("\n")

    
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
    
    def run(self, sentence: List[str]):
        self._generate_graph(sentence)
        self._generate_transition_matrix()
        self._viterbi()
        del self.graph
        del self.trans


def bi_gram_generator(input_path: str, output: str):
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
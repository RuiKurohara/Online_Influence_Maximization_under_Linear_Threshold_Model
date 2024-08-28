import numpy as np
from itertools import combinations
import datetime
import diffusion
from heapdict import heapdict

class Node(object):
    def __init__(self, node):
        self.node = node
        self.mg1 = 0
        self.prev_best = None
        self.mg2 = 0
        self.flag = None
        self.list_index = 0

def celfpp(graph, diffuse, k):
    S = set()
    # Note that heapdict is min heap and hence add negative priorities for
    # it to work.
    Q = heapdict()
    last_seed = None
    cur_best = None
    node_data_list = []

    for node in graph.nodes:
        node_data = Node(node)
        node_data.mg1 = diffuse.diffuse_mc([node])
        node_data.prev_best = cur_best
        node_data.mg2 = diffuse.diffuse_mc([node, cur_best.node]) if cur_best else node_data.mg1
        node_data.flag = 0
        cur_best = cur_best if cur_best and cur_best.mg1 > node_data.mg1 else node_data
        graph.nodes[node]['node_data'] = node_data
        node_data_list.append(node_data)
        node_data.list_index = len(node_data_list) - 1
        Q[node_data.list_index] = - node_data.mg1

    while len(S) < k:
        node_idx, _ = Q.peekitem()
        node_data = node_data_list[node_idx]
        if node_data.flag == len(S):
            S.add(node_data.node)
            del Q[node_idx]
            last_seed = node_data
            continue
        elif node_data.prev_best == last_seed:
            node_data.mg1 = node_data.mg2
        else:
            before = diffuse.diffuse_mc(S)
            S.add(node_data.node)
            after = diffuse.diffuse_mc(S)
            S.remove(node_data.node)
            node_data.mg1 = after - before
            node_data.prev_best = cur_best
            S.add(cur_best.node)
            before = diffuse.diffuse_mc(S)
            S.add(node_data.node)
            after = diffuse.diffuse_mc(S)
            S.remove(cur_best.node)
            if node_data.node != cur_best.node: S.remove(node_data.node)
            node_data.mg2 = after - before

        if cur_best and cur_best.mg1 < node_data.mg1:
            cur_best = node_data

        node_data.flag = len(S)
        Q[node_idx] = - node_data.mg1

    return S

# get random weight
def getNextRandomWeight(weightLowerBound, weightUpperBound):
    weightNowPos = np.random.uniform(weightLowerBound, weightUpperBound)
    return weightNowPos


### get Best S with DFS - START
def getActivateProbabiltiyByDFS(G, S, Ew, u, visitOneHot, node2Index):
    uActivateProbability = 0
    if u in S:
        return 1
    for parentEdge in G.in_edges(u):
        if visitOneHot[node2Index[parentEdge[0]]] == 0:
            visitOneHot[node2Index[parentEdge[0]]] = 1
            uActivateProbability = uActivateProbability + Ew[parentEdge] * G[parentEdge[0]][parentEdge[1]]['weight'] \
                                   * getActivateProbabiltiyByDFS(G, S, Ew, parentEdge[0], visitOneHot, node2Index)
    return uActivateProbability

"""
n=20:0.001sくらい
n=50:0.01sくらい
n=300で1.4sくらい
"""
def getSpreadSizeByProbability(G, Ew, S):
    node2Index = {}
    index = 0
    for tmp in G.nodes:
        node2Index[tmp] = index
        index += 1
    SpreadSize = len(S)
    for u in G.nodes:
        if u not in S:  # Calculate all nodes that are not seed nodes
            visitOneHot = np.zeros(G.number_of_nodes())
            SpreadSize = SpreadSize + getActivateProbabiltiyByDFS(G, S, Ew, u, visitOneHot, node2Index)
    return SpreadSize

def getDifferentSeedSpread(G, Ew, K):#！！！時間かかる犯人！！！
    BestSpreadSize = 0
    BestSeedSet = []
    BestSeedSet=celfpp(G,diffusion,K)
    BestSpreadSize=getSpreadSizeByProbability(G, Ew, BestSeedSet)
    return BestSpreadSize, BestSeedSet

def Enumerate_oracle(G, Ew, K):
    BestSpreadSize, BestSeedSet = getDifferentSeedSpread(G, Ew, K)
    return BestSeedSet
### get Best S with DFS - END

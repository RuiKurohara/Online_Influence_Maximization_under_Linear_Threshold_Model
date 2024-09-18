import numpy as np
from itertools import combinations
import datetime
#import diffusion
import networkx as nx
from tqdm.autonotebook import tqdm
from heapdict import heapdict

class Node(object):
    def __init__(self, node):
        self.node = node
        self.mg1 = 0
        self.prev_best = None
        self.mg2 = 0
        self.flag = None
        self.list_index = 0

def celfpp(graph,ew, k):
    S = set()
    # Note that heapdict is min heap and hence add negative priorities for
    # it to work.
    Q = heapdict()
    last_seed = None
    cur_best = None
    node_data_list = []

    # LinearThreshold クラスのインスタンスを作成
    lt_model = LinearThreshold(graph)

    for node in graph.nodes:
        node_data = Node(node)
        node_data.mg1 = lt_model.diffuse_mc([node],ew)
        node_data.prev_best = cur_best
        node_data.mg2 = lt_model.diffuse_mc([node, cur_best.node],ew) if cur_best else node_data.mg1
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
            before = lt_model.diffuse_mc(S,ew)
            S.add(node_data.node)
            after = lt_model.diffuse_mc(S,ew)
            S.remove(node_data.node)
            node_data.mg1 = after - before
            node_data.prev_best = cur_best
            S.add(cur_best.node)
            before = lt_model.diffuse_mc(S,ew)
            S.add(node_data.node)
            after = lt_model.diffuse_mc(S,ew)
            S.remove(cur_best.node)
            if node_data.node != cur_best.node: S.remove(node_data.node)
            node_data.mg2 = after - before

        if cur_best and cur_best.mg1 < node_data.mg1:
            cur_best = node_data

        node_data.flag = len(S)
        Q[node_idx] = - node_data.mg1

    return S

# get random weight 使わない
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
n=1:0.01sくらい
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

def getDifferentSeedSpread(G, Ew, K):
    BestSpreadSize = 0
    BestSeedSet = []
    BestSeedSet=list(celfpp(G,Ew,K))
    BestSpreadSize=getSpreadSizeByProbability(G, Ew, BestSeedSet)
    return BestSpreadSize, BestSeedSet

def Enumerate_oracle(G, Ew, K):
    BestSpreadSize, BestSeedSet = getDifferentSeedSpread(G, Ew, K)
    return BestSeedSet
### get Best S with DFS - END


##diffusion
class LinearThreshold(object):
    def __init__(self, graph):
        self.graph = graph
        self.neighborhood_fn = self.graph.neighbors if isinstance(self.graph, nx.Graph) else self.graph.predecessors
        
    def sample_node_thresholds_mc(self, mc):
            self.sampled_thresholds = np.random.uniform(size=(mc, len(self.graph.nodes)))

    def sample_node_thresholds(self, mcount):
        for idx, node in enumerate(self.graph.nodes):
            self.graph.nodes[node]['threshold'] = self.sampled_thresholds[mcount][idx]

    def diffusion_iter(self,ew):
        for node in self.graph.nodes:
            if self.graph.nodes[node]['is_active']:
                continue
            neighbors = self.neighborhood_fn(node)
            
            #weights = sum(self.graph.edges[neighbor, node]['weight'] 
            #            for neighbor in neighbors if self.graph.has_edge(neighbor, node))
            weights = 0
            for parentEdge in self.graph.in_edges(node):
                weights += ew[parentEdge] 
            #weights = sum(ew[self.graph.in_edges(node)])
            if weights > self.graph.nodes[node]['threshold']:#!!!threshould変更の必要あり？
                self.graph.nodes[node]['is_active'] = True

    def diffuse(self, act_nodes, mcount,ew):
        self.sample_node_thresholds(mcount)
        nx.set_node_attributes(self.graph, False, name='is_active')

        for node in act_nodes:
            self.graph.nodes[node]['is_active'] = True

        prev_active_nodes = set()
        active_nodes = set()
        while True:
            self.diffusion_iter(ew)
            prev_active_nodes = active_nodes
            active_nodes = set(i for i, v in self.graph.nodes(data=True) if v['is_active'])
            if active_nodes == prev_active_nodes:
                break
        self.graph.total_activated_nodes.append(len(active_nodes))

    def diffuse_mc(self, act_nodes,ew, mc=50):
        self.sample_node_thresholds_mc(mc)
        self.graph.total_activated_nodes = []
        for i in range(mc):
            self.diffuse(act_nodes, i,ew)
        return sum(self.graph.total_activated_nodes) / float(mc)

    def shapely_iter(self, act_nodes,ew):
        nx.set_node_attributes(self.graph, False, name='is_active')

        for node in act_nodes:
            self.graph.nodes[node]['is_active'] = True

        self.diffusion_iter(ew)
        active_nodes = [n for n, v in self.graph.nodes.data() if v['is_active']]
        return active_nodes

    def shapely_diffuse(self, nodes,ew, mc=50):
        self.sample_node_thresholds_mc(mc)
        for node in nodes:
            self.graph.nodes[node]['tmp'] = 0

        for c in tqdm(range(mc), desc='Shapely Monte Carlo', leave=False):
            self.sample_node_thresholds(c)
            active_nodes_with = []
            active_nodes_without = []
            for i in tqdm(range(len(nodes)), desc='Shapely Nodes', leave=False):
                if i in active_nodes_with:
                    self.graph.nodes[node]['tmp'] = 0
                    continue
                active_nodes_with = self.shapely_iter(nodes[:i+1],ew)
                active_nodes_without = self.shapely_iter(nodes[:i],ew)
                self.graph.nodes[nodes[i]]['tmp'] +=  len(active_nodes_with) - len(active_nodes_without)


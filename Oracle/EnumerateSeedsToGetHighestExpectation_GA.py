import numpy as np
from itertools import combinations
import networkx as nx

# get random weight
def getNextRandomWeight(weightLowerBound, weightUpperBound):
    return np.random.uniform(weightLowerBound, weightUpperBound)

### get Best S with DFS - START
# 再帰を用いないDFS
def getActivateProbabilityByDFS_non_recursive(G, S, Ew, u, node2Index):
    stack = [u]
    probability = 0
    visited = set()

    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        if node in S:
            return 1
        for parentEdge in G.in_edges(node):
            parent = parentEdge[0]
            if parent not in visited:
                stack.append(parent)
                probability += Ew[parentEdge] * G[parentEdge[0]][parentEdge[1]]['weight']

    return probability

def getSpreadSizeByProbability(G, Ew, S):
    node2Index = {node: idx for idx, node in enumerate(G.nodes)}
    SpreadSize = len(S)
    for u in G.nodes:
        if u not in S:  # Calculate all nodes that are not seed nodes
            SpreadSize += getActivateProbabilityByDFS_non_recursive(G, S, Ew, u, node2Index)
    return SpreadSize

def heuristic_seed_selection(G, Ew, K):
    node_influence = {node: getSpreadSizeByProbability(G, Ew, [node]) for node in G.nodes}
    selected_seeds = []
    for _ in range(K):
        best_node = max(node_influence, key=node_influence.get)
        selected_seeds.append(best_node)
        del node_influence[best_node]
        for node in node_influence:
            node_influence[node] = getSpreadSizeByProbability(G, Ew, selected_seeds + [node])
    return selected_seeds

def Enumerate_oracle(G, Ew, K):
    BestSeedSet = heuristic_seed_selection(G, Ew, K)
    BestSpreadSize = getSpreadSizeByProbability(G, Ew, BestSeedSet)
    return BestSeedSet, BestSpreadSize
### get Best S with DFS - END

# 関数の使用例
def example_usage():
    G = nx.DiGraph()
    # ノードとエッジをGに追加
    Ew = {edge: getNextRandomWeight(0.1, 0.9) for edge in G.edges}
    K = 3  # シードの数
    best_seeds, best_spread = Enumerate_oracle(G, Ew, K)
    print(f"Best seeds: {best_seeds}, Spread size: {best_spread}")

# example_usage()を呼び出してテストすることができます
# example_usage()

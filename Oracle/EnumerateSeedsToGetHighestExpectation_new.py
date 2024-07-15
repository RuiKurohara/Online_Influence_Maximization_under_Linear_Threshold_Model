import numpy as np
from itertools import combinations

# get random weight
def getNextRandomWeight(weightLowerBound, weightUpperBound):
    weightNowPos = np.random.uniform(weightLowerBound, weightUpperBound)
    return weightNowPos

### get Best S with DFS - START
#再起を用いないDFS
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
            if node2Index[parent] not in visited:
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

def getDifferentSeedSpread(G, Ew, K):
    BestSpreadSize = 0
    BestSeedSet = []
    for seedCombination in combinations(G.nodes, K):
        tmpSpreadSize = getSpreadSizeByProbability(G, Ew, seedCombination)
        if tmpSpreadSize > BestSpreadSize:
            BestSpreadSize = tmpSpreadSize
            BestSeedSet = list(seedCombination)
    return BestSpreadSize, BestSeedSet

def Enumerate_oracle(G, Ew, K):
    BestSpreadSize, BestSeedSet = getDifferentSeedSpread(G, Ew, K)
    return BestSeedSet
### get Best S with DFS - END

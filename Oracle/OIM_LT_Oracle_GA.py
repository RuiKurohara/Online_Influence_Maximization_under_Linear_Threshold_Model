import copy
import random
import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import inv as sparse_inv
from scipy.sparse.linalg import spsolve

# get Weigh one by one
def getNextWeight(weightNowPos, initWeight, weightUpperBound, sideLength):
    traverseEnd = 0
    for indexOfWeight in range(len(weightNowPos) + 1):
        if indexOfWeight == len(weightNowPos):
            traverseEnd = 1
            break
        if weightNowPos[indexOfWeight] + sideLength > weightUpperBound[indexOfWeight]:
            continue
        else:
            weightNowPos[indexOfWeight] = weightNowPos[indexOfWeight] + sideLength
            for underIndex in range(indexOfWeight):
                weightNowPos[underIndex] = initWeight[underIndex]
            break
    return weightNowPos, traverseEnd

# get random weight
def getNextRandomWeight(weightLowerBound, weightUpperBound):
    weightNowPos = np.random.uniform(weightLowerBound, weightUpperBound)
    return weightNowPos

def IMLinUCB_Oracle(V, b, c, epsilon, IM_oracle, IM_cal_reward, K, G, edge2Index, sampleStrategy="RandomGenerate", scaleGaussianRatio=1, seed_set=None):
    sideLength = (2 / np.sqrt(3)) * epsilon
    V_sparse = csc_matrix(V)
    invV = sparse_inv(V_sparse)
    invVb = spsolve(V_sparse, b)
    weightLowerBound = np.zeros(b.shape)
    weightUpperBound = np.ones(b.shape)

    initWeight = weightLowerBound + sideLength / 2  # init center
    weightNowPos = copy.deepcopy(initWeight).flatten()

    BestS = []
    BestReward = -1
    BestEwEstimated = {}

    i = 0
    indexOfGaussianPrioritySample = 0
    while True:
        if sampleStrategy == "RandomGenerate":
            sampleSize = 100
            weightNowPos = getNextRandomWeight(weightLowerBound.flatten(), weightUpperBound.flatten())
            if i == sampleSize:
                break
        elif sampleStrategy == "RatioSample":
            weightNowPos, traverseEnd = getNextWeight(weightNowPos, initWeight.flatten(), weightUpperBound.flatten(), sideLength)
            if traverseEnd:
                break
            sampleRatio = 10
            choiceAns = random.choice(range(sampleRatio))
            if choiceAns == 0:
                continue
        elif sampleStrategy == "GaussianPrioritySample":
            weightAveragePos = invVb
            correlation = invV * scaleGaussianRatio
            weightNowPos = np.random.multivariate_normal(weightAveragePos.flatten(), correlation.toarray()).flatten()
            weightNowPos = np.clip(weightNowPos, 0, 1)
            sampleSize = 10
            if indexOfGaussianPrioritySample == sampleSize:
                break
            indexOfGaussianPrioritySample += 1
        else:
            weightNowPos, traverseEnd = getNextWeight(weightNowPos, initWeight.flatten(), weightUpperBound.flatten(), sideLength)
            if traverseEnd:
                break

        if sampleStrategy == "GaussianPrioritySample" or (weightNowPos-invVb.flatten()).T.dot(V_sparse).dot(weightNowPos-invVb.flatten()) <= c:
            i += 1
            EwEstimated = {}
            for edge in G.in_edges():
                indexEdge = edge2Index[edge]
                EwEstimated[(edge[0], edge[1])] = weightNowPos[indexEdge] / G[edge[0]][edge[1]]['weight']

            # Use the provided seed_set if available, otherwise, find the best seed set
            if seed_set is not None:
                S = seed_set
            else:
                S = IM_oracle(G, EwEstimated, K)

            SpreadSize = IM_cal_reward(G, EwEstimated, S)

            if SpreadSize > BestReward:
                BestReward = SpreadSize
                BestS = S
                BestEwEstimated = copy.deepcopy(EwEstimated)

    return BestS, BestEwEstimated

import numpy as np
import math
import DataPreProcessing.copyGraph_ramdomWeight

class OIM_AETC_Algorithm:
    def __init__(self, G, EwTrue, seedSize, oracle, iterationTime, a):
        # アルゴリズムのパラメータ初期化
        #self.G = G
        #self.EwTrue = EwTrue  # 正確なエッジの重み
        self.G,  self.EwTrue = DataPreProcessing.copyGraph_ramdomWeight.copy_G_random(G)
        self.seedSize = seedSize  # アームの数（シードセットのサイズ）
        self.iterationTime = iterationTime  # 合計イテレーション
        self.oracle = oracle
        self.lossList = []
        self.iterCounter = 1  # 現在のイテレーション数
        self.isFirst_commit = True
        self.estimated_S ={}
        self.skipCounter = 0 #必要数カウントしたら飛ばす用
        self.initial_explore = 1
        self.a = a#ハイパーパラメータ 初期実行
        #self.b = b#ハイパーパラメータ　初期実行回数
        #self.c = c#ハイパーパラメータ　最高実行回数

        # ノードインデックスのマッピング
        self.index2Node = []
        for v in self.G.nodes():
            self.index2Node.append(v)

    
        self.XactivatedCounter = {}
        self.sum_XactivatedCounter = 0
        self.EwHat = {}
        for edge in self.G.in_edges():
            self.XactivatedCounter[edge] = 0
            self.EwHat[edge] = 0

        # 信頼パラメータ（探索のためのパラメータ）
        self.pulls = {node: 0 for node in self.G.nodes()}
        self.estimated_mean = {node: float('-inf') for node in self.G.nodes()}
        self.estimated_mean_st = {node: float('-inf') for node in self.G.nodes()}
        self.max_estimated_mean = float('-inf')
        self.upper_bound = {node: float('inf') for node in self.G.nodes()}
        self.lower_bound = {node: float('-inf') for node in self.G.nodes()}
        self.end_explore = {node: False for node in self.G.nodes()}#終了条件　すべてが負の数になったら終了

        # 探索フェーズが終了したかどうかを示すフラグ
        self.explore_phase = True
        #探索フェーズが終了したときの反復回数
        self.endCount = 0

    def decide(self):
        # S = []
        #budget×node数より小さいときは探索
        if self.iterCounter < self.a*self.G.number_of_nodes():
            uToLearning = self.index2Node[self.iterCounter % self.G.number_of_nodes()]
            S = [uToLearning]  # one node as seed set
        #
        elif self.isFirst_commit:
            S = self.oracle(self.G, self.EwHat, self.seedSize)#時間かかる
            self.estimated_S = S
            self.isFirst_commit = False
        else:
            S = self.estimated_S
        norm1BetweenEwEstimate_EwTrue = 0
        for u, v in self.EwTrue:
            norm1BetweenEwEstimate_EwTrue = norm1BetweenEwEstimate_EwTrue + abs(self.EwHat[(u, v)] - self.EwTrue[(u, v)])
        #print("norm1BetweenEwEstimate_EwTrue", norm1BetweenEwEstimate_EwTrue)
        self.lossList.append(norm1BetweenEwEstimate_EwTrue)
        EwEstimated = self.EwHat
        return S, EwEstimated

    def updateParameters(self, finalInfluencedNodeList, attemptingActivateInNodeDir,
                             ActivateInNodeOfFinalInfluencedNodeListDir_AMomentBefore):

        # update Algorithms parameters
        if self.iterCounter < self.a*self.G.number_of_nodes():
            uToLearning = self.index2Node[self.iterCounter % self.G.number_of_nodes()]
            # Update all outgoing edges of node u
            for edge in self.G.out_edges(uToLearning):
                v = edge[1]
                if v in finalInfluencedNodeList:
                    # Only when the initial node uToLearning affects v can it be counted in the counter
                    if len(attemptingActivateInNodeDir[v]) == 1:
                        self.XactivatedCounter[edge] += 1

        if self.iterCounter == self.a*self.G.number_of_nodes()-1:
            for edge in self.G.out_edges():
                self.EwHat[edge] = self.XactivatedCounter[edge] / self.a

        self.iterCounter += 1

    def getLoss(self):
        return np.asarray(self.lossList)
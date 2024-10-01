import numpy as np
import math

class OIM_AETC_Algorithm:
    def __init__(self, G, EwTrue, seedSize, oracle, iterationTime, delta=0.1):
        # アルゴリズムのパラメータ初期化
        self.G = G
        self.EwTrue = EwTrue  # 正確なエッジの重み
        self.seedSize = seedSize  # アームの数（シードセットのサイズ）
        self.iterationTime = iterationTime  # 合計イテレーション
        self.oracle = oracle
        self.lossList = []
        self.iterCounter = 1  # 現在のイテレーション数
        self.isFirst_commit = True
        self.estimated_S ={}
        self.skipCounter = 0 #必要数カウントしたら飛ばす用
        self.initial_explore = 1

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
        self.delta = delta
        self.pulls = {node: 0 for node in self.G.nodes()}
        self.estimated_mean = {node: float('-inf') for node in self.G.nodes()}
        self.max_estimated_mean = float('-inf')
        self.upper_bound = {node: float('inf') for node in self.G.nodes()}
        self.lower_bound = {node: float('-inf') for node in self.G.nodes()}
        self.end_explore = {node: False for node in self.G.nodes()}#終了条件　すべてが負の数になったら終了

        # 探索フェーズが終了したかどうかを示すフラグ
        self.explore_phase = True
        #探索フェーズが終了したときの反復回数
        self.endCount = 0

    def decide(self):
        if self.iterCounter < self.G.number_of_nodes() * self.initial_explore:
            uToLearning = self.index2Node[self.iterCounter % self.G.number_of_nodes()]
            S = [uToLearning]  # 探索する1つのノードをシードセットとして選択

        elif self.explore_phase:
            # 探索フェーズ: 各ノードを順番に選択し、シードセットを作成
            self.skipCounter = 0
            while True:
                uToLearning = self.index2Node[(self.iterCounter + self.skipCounter) % self.G.number_of_nodes()]
                S = [uToLearning]  # 探索する1つのノードをシードセットとして選択
                if self.end_explore[uToLearning] == False:#探索終了でなければbreak
                    break
                self.skipCounter += 1

            # 動的に探索フェーズを終了する条件
            if self.upper_bound[uToLearning] == self.estimated_mean[uToLearning]:
                self.end_explore[uToLearning] = False
            #elif math.log(self.seedSize*self.iterationTime)/pow(self.upper_bound[uToLearning]-self.estimated_mean[uToLearning],2)-self.pulls[uToLearning] - self.initial_explore < 0:
            elif self.seedSize*math.log(self.iterationTime)/pow(self.upper_bound[uToLearning]-self.estimated_mean[uToLearning],2)-self.pulls[uToLearning] - self.initial_explore < 0:
                print(self.seedSize*math.log(self.iterationTime)/pow(self.upper_bound[uToLearning]-self.estimated_mean[uToLearning],2)-self.pulls[uToLearning])
                self.end_explore[uToLearning] = True
            
            #print(self.end_explore)
            #print(all(self.end_explore.values()))
            if list(self.end_explore.values()).count(False) == 1:
                self.explore_phase = False  # 探索フェーズ終了
                self.endCount = self.iterCounter
                # 活用フェーズに移行'
            '''
            for node in self.G.nodes():
                #if (self.upper_bound[edge] - self.lower_bound[edge]) < self.delta:
                #print(self.upper_bound[edge])
                if self.upper_bound[node] < self.estimated_mean[node]:
                    #self.explore_phase = False  # 探索フェーズ終了
                    self.endCount = self.iterCounter
                    break  # 活用フェーズに移行'''

        elif not self.explore_phase and self.isFirst_commit:
            # 活用フェーズ: oracle関数を用いて最良のシードセットを決定
            S = self.oracle(self.G, self.EwHat, self.seedSize)
            self.estimated_S = S
            self.isFirst_commit = False
        
        elif not self.explore_phase and not self.isFirst_commit:
            S = self.estimated_S
        
        # 推定誤差の計算
        norm1BetweenEwEstimate_EwTrue = 0
        for u, v in self.EwTrue:
            norm1BetweenEwEstimate_EwTrue += abs(self.EwHat[(u, v)] - self.EwTrue[(u, v)])
        print("norm1BetweenEwEstimate_EwTrue", norm1BetweenEwEstimate_EwTrue)
        self.lossList.append(norm1BetweenEwEstimate_EwTrue)

        # 現在の推定値を返す
        EwEstimated = self.EwHat
        return S, EwEstimated

    def updateParameters(self, finalInfluencedNodeList, attemptingActivateInNodeDir,
                         ActivateInNodeOfFinalInfluencedNodeListDir_AMomentBefore):
        # 探索フェーズ中のパラメータ更新
        if self.explore_phase:
            uToLearning = self.index2Node[self.iterCounter % self.G.number_of_nodes()]
            self.pulls[uToLearning] += 1
            # ノードuの全てのアウトゴーイングエッジの更新
            for edge in self.G.out_edges(uToLearning):
                v = edge[1]
                if v in finalInfluencedNodeList:
                    # uからの影響がvに及んでいる場合にカウンタを更新
                    if len(attemptingActivateInNodeDir[v]) == 1:
                        self.XactivatedCounter[edge] += 1
                        self.sum_XactivatedCounter += 1
            
            # エッジの上限・下限の更新
            estimatade_sum=0
            for edge in self.G.out_edges(uToLearning):
                estimatade_sum += self.XactivatedCounter[edge]
            self.estimated_mean[uToLearning] = estimatade_sum / self.pulls[uToLearning]
            if self.estimated_mean[uToLearning] > self.max_estimated_mean:
                self.max_estimated_mean = self.estimated_mean[uToLearning]
            #self.upper_bound[uToLearning] = np.sqrt((2 * np.log(self.iterCounter)) / self.pulls[uToLearning])
            #self.upper_bound[uToLearning] =self.sum_XactivatedCounter /sum(self.pulls)
            self.upper_bound[uToLearning] = self.max_estimated_mean
            print(self.estimated_mean[uToLearning])
            print(self.upper_bound[uToLearning])
            
        # 探索が終了したら最良のエッジの推定値を確定
        if not self.explore_phase:
            for edge in self.G.out_edges():
                self.EwHat[edge] = self.XactivatedCounter[edge] / (self.endCount/self.G.number_of_nodes())

        self.iterCounter += 1

    def getLoss(self):
        return np.asarray(self.lossList)

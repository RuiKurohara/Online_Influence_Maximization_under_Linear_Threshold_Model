import numpy as np

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

        # ノードインデックスのマッピング
        self.index2Node = []
        for v in self.G.nodes():
            self.index2Node.append(v)

    
        self.XactivatedCounter = {}
        self.EwHat = {}
        for edge in self.G.in_edges():
            self.XactivatedCounter[edge] = 0
            self.EwHat[edge] = 0

        # 信頼パラメータ（探索のためのパラメータ）
        self.delta = delta
        self.pulls = {node: 0 for node in self.G.nodes()}
        self.estimated_mean = {node: float('-inf') for node in self.G.nodes()}
        self.upper_bound = {node: float('inf') for node in self.G.nodes()}
        self.lower_bound = {node: float('-inf') for node in self.G.nodes()}

        # 探索フェーズが終了したかどうかを示すフラグ
        self.explore_phase = True
        #探索フェーズが終了したときの反復回数
        self.endCount = 0

    def decide(self):
        if self.iterCounter < self.G.number_of_nodes():
            uToLearning = self.index2Node[self.iterCounter % self.G.number_of_nodes()]
            S = [uToLearning]  # 探索する1つのノードをシードセットとして選択
        elif self.explore_phase:
            # 探索フェーズ: 各ノードを順番に選択し、シードセットを作成
            uToLearning = self.index2Node[self.iterCounter % self.G.number_of_nodes()]
            S = [uToLearning]  # 探索する1つのノードをシードセットとして選択

            # 動的に探索フェーズを終了する条件
            
            for node in self.G.nodes():
                #if (self.upper_bound[edge] - self.lower_bound[edge]) < self.delta:
                #print(self.upper_bound[edge])
                if self.upper_bound[node] < self.estimated_mean[node]:
                    #self.explore_phase = False  # 探索フェーズ終了
                    self.endCount = self.iterCounter
                    break  # 活用フェーズに移行

        elif not self.explore_phase:
            # 活用フェーズ: oracle関数を用いて最良のシードセットを決定
            S = self.oracle(self.G, self.EwHat, self.seedSize)
        
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
            
            # エッジの上限・下限の更新
            estimatade_sum=0
            for edge in self.G.out_edges(uToLearning):
                estimatade_sum += self.XactivatedCounter[edge]
            self.estimated_mean[uToLearning] = estimatade_sum / self.pulls[uToLearning]#値大き目調整必要
            #print(self.iterCounter)
            #print(self.pulls[edge])
            self.upper_bound[uToLearning] = np.sqrt((2 * np.log(self.iterCounter)) / self.pulls[uToLearning])
            print(self.estimated_mean[uToLearning])
            print(self.upper_bound[uToLearning])
            
        # 探索が終了したら最良のエッジの推定値を確定
        if not self.explore_phase:
            for edge in self.G.out_edges():
                self.EwHat[edge] = self.XactivatedCounter[edge] / (self.endCount/self.G.number_of_nodes())

        self.iterCounter += 1

    def getLoss(self):
        return np.asarray(self.lossList)

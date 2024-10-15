import os
import pickle
import argparse
import time
import random
import numpy as np
import datetime
import optuna
import matplotlib.pyplot as plt
import DataPreProcessing.copyGraph
import DataPreProcessing.makeWSGraph
import LT.LT
import DataPreProcessing
import Oracle.EnumerateSeedsToGetHighestExpectation
import Oracle.BinaryOracle
import Oracle.EnumerateSeedsToGetHighestExpectation_GA
import Oracle.EnumerateSeedsToGetHighestExpectation_new
import Oracle.diffusion
from Tool.create_save_path import *
from BanditAlg.OIM_ETC import OIM_ETC_Algorithm
from BanditAlg.OIM_ETC_light import OIM_ETC_Algorithm_light
from BanditAlg.OIM_AETC_simple import OIM_AETC_Algorithm
from BanditAlg.IMLinUCB_LT import IMLinUCB_LT_Algorithm as IMLinUCB_LT_Algorithm_TS
from BanditAlg.IMLinUCB_LT_new import IMLinUCB_LT_Algorithm as IMLinUCB_LT_Algorithm_TS_new  # 新しいアルゴリズム
from BanditAlg.IMLinUCB_LT_GA import IMLinUCB_LT_Algorithm_GA as IMLinUCB_LT_Algorithm_TS_GA  # GAアルゴリズム
from BanditAlg.IMLinUCB_LT_little_V_binary_2d import IMLinUCB_LT_Algorithm as IMLinUCB_LT_Algorithm_2d

global start
class simulateOnlineData:
    def __init__(self, G, EwTrue, lv, seed_size, oracle, calculate_exact_spreadsize, iterationTime, dataset, RandomSeed):
        self.G = G
        self.EwTrue = EwTrue  # True weight
        self.lv = lv #ノードの閾値
        self.seed_size = seed_size
        self.oracle = oracle
        self.calculate_exact_spreadsize = calculate_exact_spreadsize
        self.iterationTime = iterationTime
        self.dataset = dataset
        self.RandomSeed = RandomSeed
        self.startTime = datetime.datetime.now()
        self.AlgReward = {}
        self.AlgLoss = {}
        self.AlgRegret = {}

    def runAlgorithms(self, algorithms):
        self.tim_ = []
        for alg_name, alg in list(algorithms.items()):
            self.AlgReward[alg_name] = []
            self.AlgLoss[alg_name] = []
            self.AlgRegret[alg_name] = []

        self.resultRecord()
        BestSeedSet = self.oracle(self.G, self.EwTrue, self.seed_size)#ここだけは厳密解の必要あり？
        #BestSeedSet = Oracle.EnumerateSeedsToGetHighestExpectation.Enumerate_oracle(self.G, self.EwTrue, self.seed_size)
        BestSpreadSize = self.calculate_exact_spreadsize(self.G, self.EwTrue, BestSeedSet)

        for iter_ in range(self.iterationTime):
            for alg_name, alg in list(algorithms.items()):
                # 1. use Online Algs to decide seed
                start_decide=datetime.datetime.now()
                print("\n1. Get seed with Online Algs")
                S, EwEstimated = alg.decide()
                print("seed set", S)  # list
                print("decide_time",datetime.datetime.now()-start_decide)

                # 2. get live_edge/node from LT
                # observe edge level feedback 神様視点
                start_sim=datetime.datetime.now()
                print("2. Simulate Influence Spreading on LT")
                rewardTrue, finalInfluencedNodeList, workedInNodeList, attemptingActivateInNodeDir, ActivateInNodeOfFinalInfluencedNodeListDir_AMomentBefore = LT.LT.runLT_NodeFeedback(
                    G, S, EwTrue, lv)
                print("Simulated Result: size is", rewardTrue)
                print("sim_time",datetime.datetime.now()-start_sim)

                # 2 get Reward　観測できる
                start_reward=datetime.datetime.now()
                print("2. Get Expectation Reward of Algs Seeds")
                reward = self.calculate_exact_spreadsize(self.G, self.EwTrue, S)
                print("Expected Reward: size is", reward)
                print("reward_time",datetime.datetime.now()-start_reward)

                # 3. Update parameters A b
                start_update=datetime.datetime.now()
                print("3. Update parameters A b")
                alg.updateParameters(finalInfluencedNodeList, attemptingActivateInNodeDir,
                                     ActivateInNodeOfFinalInfluencedNodeListDir_AMomentBefore)
                print("update_time",datetime.datetime.now()-start_update)

                # 4. Record results
                self.AlgReward[alg_name].append(reward)
                self.AlgRegret[alg_name].append(BestSpreadSize - reward)
                self.AlgLoss[alg_name].append(alg.getLoss()[-1])

            self.resultRecord(iter_)
            self.recordBestSeedSet(BestSeedSet, reward)
            self.recordExcutionTime()
        print("No", iter_, ":Average Oracle Reward", BestSpreadSize)
        print("Best Seed Set", BestSeedSet)#表示しても意味ない？求めたいのは厳密解のシードセットではなく推定解のシードセット

    def runAlgorithms_train(self, algorithms):
        self.tim_ = []
        for alg_name, alg in list(algorithms.items()):
            self.AlgReward[alg_name] = []
            self.AlgLoss[alg_name] = []
            self.AlgRegret[alg_name] = []
        reward_avarege = 0

        #self.resultRecord()
        BestSeedSet = self.oracle(self.G, self.EwTrue, self.seed_size)#ここだけは厳密解の必要あり？
        #BestSeedSet = Oracle.EnumerateSeedsToGetHighestExpectation.Enumerate_oracle(self.G, self.EwTrue, self.seed_size)
        BestSpreadSize = self.calculate_exact_spreadsize(self.G, self.EwTrue, BestSeedSet)

        for iter_ in range(self.iterationTime):
            for alg_name, alg in list(algorithms.items()):
                # 1. use Online Algs to decide seed
                #start_decide=datetime.datetime.now()
                #print("\n1. Get seed with Online Algs")
                S, EwEstimated = alg.decide()
                #print("seed set", S)  # list
                #print("decide_time",datetime.datetime.now()-start_decide)

                # 2. get live_edge/node from LT
                # observe edge level feedback 神様視点
                #start_sim=datetime.datetime.now()
                #print("2. Simulate Influence Spreading on LT")
                rewardTrue, finalInfluencedNodeList, workedInNodeList, attemptingActivateInNodeDir, ActivateInNodeOfFinalInfluencedNodeListDir_AMomentBefore = LT.LT.runLT_NodeFeedback(
                    G, S, EwTrue, lv)
                """
                rewardTrue, finalInfluencedNodeList, workedInNodeList, attemptingActivateInNodeDir, ActivateInNodeOfFinalInfluencedNodeListDir_AMomentBefore = LT.LT.runLT_NodeFeedback_train(
                    G, S, EwTrue, lv)
                """
                #print("Simulated Result: size is", rewardTrue)
                #print("sim_time",datetime.datetime.now()-start_sim)

                # 2 get Reward　観測できる
                start_reward=datetime.datetime.now()
                #print("2. Get Expectation Reward of Algs Seeds")
                reward = self.calculate_exact_spreadsize(self.G, self.EwTrue, S)
                #print("Expected Reward: size is", reward)
                reward_avarege += reward
                #print(reward)
                #print("reward_time",datetime.datetime.now()-start_reward)

                # 3. Update parameters A b
                #start_update=datetime.datetime.now()
                #print("3. Update parameters A b")
                alg.updateParameters(finalInfluencedNodeList, attemptingActivateInNodeDir,
                                     ActivateInNodeOfFinalInfluencedNodeListDir_AMomentBefore)
                #print("update_time",datetime.datetime.now()-start_update)
        reward_avarege = reward_avarege/self.iterationTime
        reward_percent = reward_avarege/BestSpreadSize

        return reward_percent

    def resultRecord(self, iter_=None):
        if iter_ is None:
            # Initialize the header
            timeRun = self.startTime.strftime('_%m_%d_%H_%M_%S')
            fileSig = '_seedsize' + str(self.seed_size) + '_iter' + str(self.iterationTime) + '_' + self.dataset + "_RandomSeed" + str(self.RandomSeed)

            self.filenameWriteReward = os.path.join(save_address, 'Reward/Reward' + timeRun + fileSig + '.csv')
            os.makedirs(os.path.dirname(self.filenameWriteReward), exist_ok=True)
            with open(self.filenameWriteReward, 'w') as f:
                f.write('Time(Iteration)')
                f.write(',' + ','.join([str(alg_name) for alg_name in algorithms.keys()]))
                f.write('\n')

            self.filenameWriteParameterLoss = os.path.join(save_address, 'ParameterLoss/Lossweight' + timeRun + fileSig + '.csv')
            os.makedirs(os.path.dirname(self.filenameWriteParameterLoss), exist_ok=True)
            with open(self.filenameWriteParameterLoss, 'w') as f:
                f.write('Time(Iteration)')
                f.write(',' + ','.join([str(alg_name) for alg_name in algorithms.keys()]))
                f.write('\n')

            self.filenameWriteRegret = os.path.join(save_address, 'Regret/Regret' + timeRun + fileSig + '.csv')
            os.makedirs(os.path.dirname(self.filenameWriteRegret), exist_ok=True)
            with open(self.filenameWriteRegret, 'w') as f:
                f.write('Time(Iteration)')
                f.write(',' + ','.join([str(alg_name) for alg_name in algorithms.keys()]))
                f.write('\n')

            self.filenameWriteElapsedTime = os.path.join(save_address, 'ElapsedTime/ElapsedTime' + timeRun + fileSig + '.csv')
            os.makedirs(os.path.dirname(self.filenameWriteElapsedTime), exist_ok=True)
            with open(self.filenameWriteElapsedTime, 'w') as f:
                f.write('Time(Iteration),ElapsedTime\n')

        else:
            elapsed_time = datetime.datetime.now() - self.startTime
            print("Iteration %d" % iter_, " Elapsed time", elapsed_time)
            self.tim_.append(iter_)
            with open(self.filenameWriteReward, 'a+') as f:
                f.write(str(iter_))
                f.write(
                    ',' + ','.join([str(self.AlgReward[alg_name][-1]) for alg_name in algorithms.keys()]))  # Record the last number
                f.write('\n')

            with open(self.filenameWriteParameterLoss, 'a+') as f:
                f.write(str(iter_))
                f.write(
                    ',' + ','.join([str(self.AlgLoss[alg_name][-1]) for alg_name in algorithms.keys()]))
                f.write('\n')

            with open(self.filenameWriteRegret, 'a+') as f:
                f.write(str(iter_))
                f.write(
                    ',' + ','.join([str(self.AlgRegret[alg_name][-1]) for alg_name in algorithms.keys()]))
                f.write('\n')

            with open(self.filenameWriteElapsedTime, 'a+') as f:
                f.write(f'{iter_},{elapsed_time}\n')

    def recordBestSeedSet(self, BestSeedSet, reward):
        timeRun = self.startTime.strftime('_%m_%d_%H_%M_%S')
        fileSig = '_seedsize' + str(self.seed_size) + '_iter' + str(self.iterationTime) + '_' + self.dataset + "_RandomSeed" + str(self.RandomSeed)
        
        filenameBestSeedSet = os.path.join(save_address, 'BestSeedSet/BestSeedSet' + timeRun + fileSig + '.csv')
        os.makedirs(os.path.dirname(filenameBestSeedSet), exist_ok=True)
        
        with open(filenameBestSeedSet, 'w') as f:
            f.write('BestSeedSet,BestSpreadSize\n')
            f.write(f'{BestSeedSet},{reward}\n')
            f.write('Hyper Parameter Score\n')
            f.write(f'{study.best_params},{study.best_value}\n')
    
    def recordExcutionTime(self):
        timeRun = self.startTime.strftime('_%m_%d_%H_%M_%S')
        fileSig = '_ExcutionTime' + str(self.seed_size) + '_iter' + str(self.iterationTime) + '_' + self.dataset + "_RandomSeed" + str(self.RandomSeed)
        
        filenameTime = os.path.join(save_address, 'ExcutionTime/ExcutionTime' + timeRun + fileSig + '.csv')
        os.makedirs(os.path.dirname(filenameTime), exist_ok=True)
        
        with open(filenameTime, 'w') as f:
            f.write('Excution Time\n')
            f.write(f'{datetime.datetime.now() - start}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--is_bipartite', action='store_true', default=False)
    parser.add_argument('--use_new_algorithm', action='store_true', default=False)  # 作成したアルゴリズムを使用
    parser.add_argument('--use_GA_algorithm', action='store_true', default=False)  # GAアルゴリズムを使用
    parser.add_argument("--seed_size", type=int, default=1, help="")
    parser.add_argument("--iterationTimes", type=int, default=50, help="")
    parser.add_argument("--save_address", type=str, default="SimulationResults/gaussian_9_ER", help="")
    parser.add_argument("--dataset_name", type=str, default="", help="")
    parser.add_argument("--G_address", type=str, default="Datasets/ER_node9_p_0.2.G", help="")
    parser.add_argument("--weight_address", type=str, default="Datasets/ER_node9_p_0.2EWTrue.dic", help="")
    parser.add_argument("--LinUCB_algs_name", type=str, default="LT-LinUCB", help="")
    parser.add_argument("--budgetList", nargs='*', default=[2, 5, 10, 20, 50,100, 200])
    args = parser.parse_args()
    start = datetime.datetime.now()#プログラムの総実行時間を調べる
    budgetList = []
    for budget_each in args.budgetList:
        budgetList.append(int(budget_each))
    print(budgetList)
    print(args.budgetList)

    if args.use_new_algorithm:
        # 新しいアルゴリズムを使用
        oracle = Oracle.EnumerateSeedsToGetHighestExpectation_new.Enumerate_oracle
        calculate_exact_spreadsize = Oracle.EnumerateSeedsToGetHighestExpectation_new.getSpreadSizeByProbability
        IMLinUCB_LT_Algorithm = IMLinUCB_LT_Algorithm_TS_new

    elif args.use_GA_algorithm:
        oracle = Oracle.EnumerateSeedsToGetHighestExpectation_GA.Enumerate_oracle
        calculate_exact_spreadsize = Oracle.EnumerateSeedsToGetHighestExpectation_GA.getSpreadSizeByProbability
        IMLinUCB_LT_Algorithm = IMLinUCB_LT_Algorithm_TS_GA

    elif args.is_bipartite:
        # ネットワークが二部グラフのとき
        oracle = Oracle.BinaryOracle.getOracleOfBinary  # 二部グラフ用オラクル
        calculate_exact_spreadsize = Oracle.BinaryOracle.getSpreadOfBinary
        IMLinUCB_LT_Algorithm = IMLinUCB_LT_Algorithm_2d  # 二部グラフ用LinUCB

    else:
        oracle = Oracle.EnumerateSeedsToGetHighestExpectation.Enumerate_oracle
        calculate_exact_spreadsize = Oracle.EnumerateSeedsToGetHighestExpectation.getSpreadSizeByProbability
        IMLinUCB_LT_Algorithm = IMLinUCB_LT_Algorithm_TS

    seed_size = args.seed_size
    iterationTimes = args.iterationTimes
    save_address = args.save_address
    create_save_path(save_address)
    dataset_name = args.dataset_name
    G = pickle.load(open(args.G_address, 'rb'), encoding='latin1')
    EwTrue = pickle.load(open(args.weight_address, 'rb'), encoding='latin1')
    lv = dict()  # threshold for nodes
    for u in G.nodes:
        lv[u] = random.random()
    LinUCB_algs_name = args.LinUCB_algs_name
    sigma = 1
    delta = 0.1

    # Fix numpy seed for reproducibility
    RandomSeed = int(time.time() * 100) % 399
    print("RandomSeed = %d" % RandomSeed)
    np.random.seed(RandomSeed)
    random.seed(RandomSeed)

    #optuna
    #train_G, train_EW = DataPreProcessing.copyGraph.copy_G_random(G)
    train_G, train_EW = DataPreProcessing.copyGraph.copy_G_edgenum(G)
    simExperiment_train = simulateOnlineData(train_G, train_EW, lv, seed_size, oracle, calculate_exact_spreadsize, iterationTimes, dataset_name, RandomSeed)
    #simExperiment_train = simulateOnlineData(G, EwTrue, lv, seed_size, oracle, calculate_exact_spreadsize, iterationTimes, dataset_name, RandomSeed)
    # Optunaでa, bを最適化
    a_max = int((iterationTimes/len(G.nodes))*0.3)#反復回数の30%探索が最大探索回数
    import optuna

    def objective(trial):
        a = trial.suggest_int('a', 1, a_max)  # ハイパーパラメータaの範囲を定義
        algorithms_train = {'AETC_train': OIM_AETC_Algorithm(train_G, train_EW, seed_size, oracle, iterationTimes, a)}

        # 再試行の回数を設定（例：5回）
        n_repeats = 5

        # 評価結果を格納するリスト
        results = []

        # 再試行を行い、その結果を格納
        for i in range(n_repeats):
            result = simExperiment_train.runAlgorithms_train(algorithms_train)  # メトリクスを取得
            results.append(result)

            # 途中結果を Optuna に報告し、プルーニングを行う
            trial.report(np.mean(results), i)

            # 早期終了の条件を満たす場合、プルーニングを実施
            if trial.should_prune():
                raise optuna.TrialPruned()

        # 結果の平均を取って返す
        return np.mean(results)

    # HyperbandPruner を使用する
    pruner = optuna.pruners.HyperbandPruner()

    # Optuna の Study を作成し、HyperbandPruner を指定
    study = optuna.create_study(direction='maximize', pruner=pruner)

    # 最適化を実行
    study.optimize(objective, n_trials=50)

    # 結果の確認
    print(f"Best parameter: {study.best_params}")
    print(f"Best objective value: {study.best_value}")
    save_address_optuna =  os.path.join(save_address, 'optuna_plot')
    # ディレクトリが存在しない場合は作成
    os.makedirs(save_address_optuna, exist_ok=True)
    # 現在の時間を取得し、ファイル名に使用
    current_time = current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"optuna_plot_{current_time}.png"
    plt.plot([trial.value for trial in study.trials])
    plt.grid()
    # フルパスを作成して画像を保存
    full_path = os.path.join(save_address_optuna, file_name)
    plt.savefig(full_path)  # 保存するファイルパスを指定


    print("Num of nodes", len(G.nodes))
    print("Num of edges", len(G.in_edges))

    simExperiment = simulateOnlineData(G, EwTrue, lv, seed_size, oracle, calculate_exact_spreadsize, iterationTimes,
                                       dataset_name, RandomSeed)
    algorithms = {}

    """
    algorithms[LinUCB_algs_name] = IMLinUCB_LT_Algorithm(G, EwTrue, seed_size, iterationTimes, sigma, delta, oracle,
                                                         calculate_exact_spreadsize)
    """
    
    for budgetTime in budgetList:
        algorithms['budget=' + str(budgetTime)] = OIM_ETC_Algorithm_light(G, EwTrue, seed_size, oracle, iterationTimes,
                                                                    budgetTime=budgetTime)
    #ETCとの比較しないときコメントアウト
    algorithms['AETC'] = OIM_ETC_Algorithm_light(G, EwTrue, seed_size, oracle, iterationTimes,study.best_params['a'])
    #algorithms['AETC'] = OIM_AETC_Algorithm(G, EwTrue, seed_size, oracle, iterationTimes,study.best_params['a'])
    #algorithms['AETC_train'] = OIM_AETC_Algorithm(train_G, train_EW, seed_size, oracle, iterationTimes,study.best_params['a'])
    simExperiment.runAlgorithms(algorithms=algorithms)
    

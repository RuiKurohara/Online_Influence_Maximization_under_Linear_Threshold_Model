import random
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms

# 評価関数 (ノイズを含む)
def eval_func(individual):
    x = individual[0]
    noise = random.gauss(0, 5)
    return -(x - 3)**2 + noise,  # 最大化を行う

# GA設定
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_int", random.randint, 0, 50)  # 範囲指定 (整数)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, 1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# toolbox.register("mate", tools.cxOnePoint)  # 交叉操作は削除
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", eval_func)

# パラメータ設定
population = toolbox.population(n=10)
ngen = 10  # 世代数
mutpb = 0.2  # 突然変異確率

# 世代ごとの最良個体のリスト
best_ind_per_gen = []

# 最適化実行
for gen in range(ngen):
    # 突然変異のみ適用
    offspring = list(map(toolbox.clone, population))
    for mutant in offspring:
        if random.random() < mutpb:
            toolbox.mutate(mutant)
            del mutant.fitness.values  # フィットネスを無効に

    # 子孫の評価
    fits = list(map(toolbox.evaluate, offspring))
    for fit, ind in zip(fits, offspring):
        ind.fitness.values = fit

    # 次世代の個体群を選択
    population = toolbox.select(offspring, k=len(population))

    # 現世代の最良個体を保存
    best_ind = tools.selBest(population, 1)[0]
    best_ind_per_gen.append(best_ind[0])

# 結果
best_ind = tools.selBest(population, 1)[0]
print("最適なx:", best_ind[0])

# 探索過程の可視化
plt.plot(best_ind_per_gen)
plt.xlabel('Generation')
plt.ylabel('Best individual (x)')
plt.title('Genetic Algorithm Optimization Progress')
plt.show()

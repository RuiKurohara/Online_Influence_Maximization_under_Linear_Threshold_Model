import numpy as np
import random
from scipy.sparse import eye, csc_matrix
from scipy.sparse.linalg import inv
import Oracle.OIM_LT_Oracle_GA as Oracle

class IMLinUCB_LT_Algorithm_GA:
    def __init__(self, G, EwTrue, seed_size, iterationTime, sigma, delta, IM_oracle, IM_cal_reward,
                 scaleTOrNot=False, scaleCRatio=1, scaleGaussianRatio=1, sampleStrategy="GaussianPrioritySample",
                 population_size=50, mutation_rate=0.1, crossover_rate=0.9, generations=100):
        self.G = G
        self.EwTrue = EwTrue
        self.seed_size = seed_size
        self.iterationTime = iterationTime
        self.iterCounter = 0

        self.IM_oracle = IM_oracle
        self.IM_cal_reward = IM_cal_reward
        self.loss_list = []

        self.V = eye(G.number_of_edges(), format='csr')
        self.b = np.zeros((G.number_of_edges(), 1))
        self.edge2Index = {}
        index = 0
        for v in self.G.nodes():
            for edge in G.in_edges(v):
                self.edge2Index[edge] = index
                index += 1

        self.scaleTOrNot = scaleTOrNot
        self.scaleCRatio = scaleCRatio
        self.scaleGaussianRatio = scaleGaussianRatio
        self.sampleStrategy = sampleStrategy

        # GA parameters
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.generations = generations

    def initialize_population(self):
        population = []
        nodes = list(self.G.nodes())
        for _ in range(self.population_size):
            individual = random.sample(nodes, self.seed_size)
            population.append(individual)
        return population

    def evaluate_population(self, population):
        fitness_scores = []
        for individual in population:
            S, EwEstimated = self.calculate_influence(individual)
            fitness_scores.append(self.calculate_fitness(S, EwEstimated))
        return fitness_scores

    def calculate_influence(self, seed_set):
        m = self.G.number_of_edges()
        n = self.G.number_of_nodes()
        if self.scaleTOrNot:
            T = self.iterCounter + 1
        else:
            T = self.iterationTime
        c = (np.sqrt(m * np.log(1 + T * n) + 2 * np.log(T * (n + 1 - self.seed_size))) + np.sqrt(n)) ** 2
        c = c * self.scaleCRatio
        epsilon = 1 / np.sqrt(self.iterationTime)

        S, EwEstimated = Oracle.IMLinUCB_Oracle(
            self.V, self.b, c, epsilon, self.IM_oracle, self.IM_cal_reward,
            self.seed_size, self.G, self.edge2Index, sampleStrategy=self.sampleStrategy,
            scaleGaussianRatio=self.scaleGaussianRatio, seed_set=seed_set
        )
        return S, EwEstimated

    def calculate_fitness(self, S, EwEstimated):
        norm1BetweenEwEstimate_EwTrue = 0
        for u, v in self.EwTrue:
            norm1BetweenEwEstimate_EwTrue += abs(EwEstimated[(u, v)] - self.EwTrue[(u, v)]) * self.G[u][v]['weight']
        loss = -norm1BetweenEwEstimate_EwTrue  # Minimize the loss
        self.loss_list.append(loss)  # ここで loss_list を更新
        return loss
    
    def select_parents(self, population, fitness_scores):
        fitness_scores = np.array(fitness_scores)
        probabilities = fitness_scores / fitness_scores.sum()
        selected_indices = np.random.choice(range(self.population_size), size=self.population_size, p=probabilities)
        return [population[i] for i in selected_indices]


    def crossover(self, parent1, parent2):
        if random.random() < self.crossover_rate:
            point = random.randint(1, self.seed_size - 1)
            child1 = parent1[:point] + parent2[point:]
            child2 = parent2[:point] + parent1[point:]
            return child1, child2
        else:
            return parent1, parent2

    def mutate(self, individual):
        if random.random() < self.mutation_rate:
            index = random.randint(0, self.seed_size - 1)
            new_gene = random.choice(list(self.G.nodes()))
            individual[index] = new_gene
        return individual

    def decide(self):
        population = self.initialize_population()
        for generation in range(self.generations):
            fitness_scores = self.evaluate_population(population)
            new_population = []
            selected_parents = self.select_parents(population, fitness_scores)
            for i in range(0, self.population_size, 2):
                parent1 = selected_parents[i]
                parent2 = selected_parents[i+1] if i+1 < self.population_size else selected_parents[0]
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                new_population.extend([child1, child2])
            population = new_population
        best_individual = population[np.argmax(fitness_scores)]
        S, EwEstimated = self.calculate_influence(best_individual)
        return S, EwEstimated

    def updateParameters(self, finalInfluencedNodeList, attemptingActivateInNodeDir,
                         attemptingActivateInNodeDir_AMomentBefore):
        for v in self.G.nodes():
            activeEdgeOnehot_v = np.zeros((self.G.number_of_edges(), 1))
            if v in finalInfluencedNodeList:
                if len(attemptingActivateInNodeDir_AMomentBefore) > 0:
                    choice = random.choice((1, 2))
                    if choice == 1:
                        for edge in self.G.in_edges(v):
                            indexOfEdge = self.edge2Index[edge]
                            if edge[0] in attemptingActivateInNodeDir[v]:
                                activeEdgeOnehot_v[indexOfEdge][0] = 1
                        y = 1
                    else:
                        for edge in self.G.in_edges(v):
                            indexOfEdge = self.edge2Index[edge]
                            if edge[0] in attemptingActivateInNodeDir_AMomentBefore[v]:
                                activeEdgeOnehot_v[indexOfEdge][0] = 1
                        y = 0
                else:
                    for edge in self.G.in_edges(v):
                        indexOfEdge = self.edge2Index[edge]
                        if edge[0] in attemptingActivateInNodeDir[v]:
                            activeEdgeOnehot_v[indexOfEdge][0] = 1
                    y = 1
            else:
                for edge in self.G.in_edges(v):
                    indexOfEdge = self.edge2Index[edge]
                    if edge[0] in attemptingActivateInNodeDir[v]:
                        activeEdgeOnehot_v[indexOfEdge][0] = 1
                y = 0

            self.V = self.V + activeEdgeOnehot_v.dot(activeEdgeOnehot_v.T)
            self.b = self.b + activeEdgeOnehot_v * y

        self.iterCounter += 1

    def getLoss(self):
        return np.asarray(self.loss_list)

# Example usage for testing
if __name__ == "__main__":
    G = nx.DiGraph()
    # Add nodes and edges to G as needed
    Ew = {edge: getNextRandomWeight(0.1, 0.9) for edge in G.edges}
    K = 3  # Number of seeds
    alg = IMLinUCB_LT_Algorithm_GA(G, Ew, K, 100, 0.1, 0.1, heuristic_seed_selection, getSpreadSizeByProbability)
    best_seeds, best_spread = alg.decide()
    print(f"Best seeds: {best_seeds}, Spread size: {best_spread}")

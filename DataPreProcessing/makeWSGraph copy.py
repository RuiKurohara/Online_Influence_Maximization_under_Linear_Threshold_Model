import networkx as nx
import random
import matplotlib.pyplot as plt
import pickle

nodesNum = 25  # ノード数
k = 4           # 各ノードが接続する近隣ノードの数（偶数である必要があります）
p = 0.1         # エッジを再配線する確率
save_graph_dir = '..//Datasets//'

def gen_WS_random(nodesNum, k, p):
    # Generate Watts-Strogatz (WS) Graph as a directed graph
    G = nx.watts_strogatz_graph(n=nodesNum, k=k, p=p, seed=None)
    
    # Convert to directed graph
    G = G.to_directed()

    # Add random weights to edges
    dir_of_edgeWeight = {}
    for edge in G.edges():
        dir_of_edgeWeight[edge] = random.uniform(0, 1)

    # Normalize edge weights for each node's in-edges
    for node in G.nodes():
        sum_of_edge_weight_V = 0
        for edge in G.in_edges(node):  # Now it works since G is directed
            sum_of_edge_weight_V += dir_of_edgeWeight[edge]
        if sum_of_edge_weight_V > 1 + 1e-4:
            for edge in G.in_edges(node):
                dir_of_edgeWeight[edge] = dir_of_edgeWeight[edge] / sum_of_edge_weight_V

    EwTrue = dir_of_edgeWeight
    return G, EwTrue

# Create the WS graph
G, EwTrue = gen_WS_random(nodesNum=nodesNum, k=k, p=p)

print(G.number_of_nodes())
print(G.number_of_edges())
print(EwTrue)

# Save the graph and edge weights
pickle.dump(G, open(save_graph_dir + "WS_node" + str(nodesNum) + '_k_' + str(k) + '_p_' + str(p) + '.G', "wb"))
pickle.dump(EwTrue, open(save_graph_dir + "WS_node" + str(nodesNum) + '_k_' + str(k) + '_p_' + str(p) + 'EWTrue.dic', "wb"))

# Visualize the graph
nx.draw(G, with_labels=True, font_weight='bold')
plt.show()

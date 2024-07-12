import igraph
import networkx as nx
import pickle
import matplotlib.pyplot as plt

save_graph_dir = '..//Datasets//'

def load_graph_from_txt(file_path):
    G = nx.DiGraph()
    with open(file_path, 'r') as file:
        for line in file:
            source, target, weight = line.strip().split()
            source = int(source)
            target = int(target)
            weight = float(weight)
            G.add_weighted_edges_from([(source, target, weight)])
    return G

# 入力ファイルパス
input_file_path = 'Instagram.txt'  # ここを入力するtxtファイルのパスに置き換えてください

G = load_graph_from_txt(input_file_path)

dir_of_edgeWeight = {}
for node in G.nodes():
    for edge in G.in_edges(node):
        dir_of_edgeWeight[edge] = G[edge[0]][edge[1]]['weight']

for node in G.nodes():
    sum_of_edge_weight_V = 0
    for edge in G.in_edges(node):
        sum_of_edge_weight_V += dir_of_edgeWeight[edge]
    if sum_of_edge_weight_V > 1+1e-4:
        for edge in G.in_edges(node):
            dir_of_edgeWeight[edge] = dir_of_edgeWeight[edge]/sum_of_edge_weight_V

EwTrue = dir_of_edgeWeight

print(G.number_of_nodes())
print(G.number_of_edges())

print(EwTrue)
pickle.dump(G, open(save_graph_dir + "graph_from_txt.G", "wb"))
pickle.dump(EwTrue, open(save_graph_dir + "graph_from_txt_EWTrue.dic", "wb"))
nx.draw(G, with_labels=True, font_weight='bold')
plt.show()
import igraph
import networkx as nx
import pickle
import matplotlib.pyplot as plt

save_graph_dir = '..//Datasets//'

def load_graph_from_txt(file_path):
    G = nx.DiGraph()
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 3:
                source, target, weight = parts
                source = int(source)
                target = int(target)
                weight = float(weight)
                G.add_weighted_edges_from([(source, target, weight)])
            else:
                print(f"Skipping invalid line: {line.strip()}")
    return G

# 入力ファイルパス
input_file_path = 'path_to_your_input_file.txt'  # ここを入力するtxtファイルのパスに置き換えてください

G = load_graph_from_txt(input_file_path)

dir_of_edgeWeight = {}
for node in G.nodes():
    for edge in G.in_edges(node):
        dir_of_edgeWeight[edge] = G[edge[0]][edge[1]]['weight']

for node in G.nodes():
    sum_of_edge_weight_V = 0
    for edge in G.in_edges(node):
        sum_of_edge_weight_V += dir_of_edgeWeight[edge]
    if sum_of_edge_weight_V > 1+1e-4:
        for edge in G.in_edges(node):
            dir_of_edgeWeight[edge] = dir_of_edgeWeight[edge]/sum_of_edge_weight_V

EwTrue = dir_of_edgeWeight

print(G.number_of_nodes())
print(G.number_of_edges())

print(EwTrue)
pickle.dump(G, open(save_graph_dir + "graph_from_txt.G", "wb"))
pickle.dump(EwTrue, open(save_graph_dir + "graph_from_txt_EWTrue.dic", "wb"))
nx.draw(G, with_labels=True, font_weight='bold')
plt.show()

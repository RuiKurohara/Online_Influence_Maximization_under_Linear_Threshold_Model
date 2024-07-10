import os
import networkx as nx
import matplotlib.pyplot as plt

def read_edges_from_file(file_path):
    edges = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.strip():
                key, value = line.strip().split(', 値: ')
                nodes = key.split(': ')[1].strip('()').split(', ')
                weight = float(value)
                edges.append((int(nodes[0]), int(nodes[1]), weight))
    return edges

def create_weighted_digraph(edges):
    G = nx.DiGraph()
    for u, v, weight in edges:
        G.add_edge(u, v, weight=weight)
    return G

def draw_weighted_digraph(G, output_path):
    pos = nx.spring_layout(G)
    weights = nx.get_edge_attributes(G, 'weight').values()
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10, arrows=True)
    #nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'weight'))
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), arrowstyle='-|>', arrowsize=10, width=[weight * 5 for weight in weights])
    plt.savefig(output_path)
    plt.clf()  # 次のグラフを描画する前に現在の図をクリア

def process_directory(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)
            edges = read_edges_from_file(file_path)
            G = create_weighted_digraph(edges)
            output_path = os.path.join(directory, filename.replace('.txt', '.png'))
            draw_weighted_digraph(G, output_path)

# メインプログラム
directory = '.'  # ここに処理したいディレクトリのパスを指定してください（現在のディレクトリの場合は '.'）
process_directory(directory)

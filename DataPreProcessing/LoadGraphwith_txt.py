import igraph
import networkx as nx
import pickle
import matplotlib.pyplot as plt
import argparse
import os
"""
txtファイルからグラフを作成
txtDatasetsにデータセットを入れてコマンドライン引数でファイル名を指定
例） python LoadGraphwith_txt.py Instagram.txt
txtファイルは
始点 終点 エッジの重み
"""
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

def main():
    parser = argparse.ArgumentParser(description='Generate a directed network from a specified txt file.')
    parser.add_argument('input_file', type=str, help='Path to the input txt file')

    args = parser.parse_args()
    input_file_path ="./txtDatasets/"+ args.input_file
    base_name = os.path.splitext(os.path.basename(input_file_path))[0]

    save_graph_dir = '..//Datasets//'
    graph_output_path = os.path.join(save_graph_dir, f"{base_name}.G")
    weight_output_path = os.path.join(save_graph_dir, f"{base_name}_EWTrue.dic")

    G = load_graph_from_txt(input_file_path)

    dir_of_edgeWeight = {}
    for node in G.nodes():
        for edge in G.in_edges(node):
            dir_of_edgeWeight[edge] = G[edge[0]][edge[1]]['weight']

    for node in G.nodes():
        sum_of_edge_weight_V = 0
        for edge in G.in_edges(node):
            sum_of_edge_weight_V += dir_of_edgeWeight[edge]
        if sum_of_edge_weight_V > 1 + 1e-4:
            for edge in G.in_edges(node):
                dir_of_edgeWeight[edge] = dir_of_edgeWeight[edge] / sum_of_edge_weight_V

    EwTrue = dir_of_edgeWeight

    print(G.number_of_nodes())
    print(G.number_of_edges())

    print(EwTrue)
    pickle.dump(G, open(graph_output_path, "wb"))
    pickle.dump(EwTrue, open(weight_output_path, "wb"))
    nx.draw(G, with_labels=True, font_weight='bold')
    plt.show()

if __name__ == '__main__':
    main()

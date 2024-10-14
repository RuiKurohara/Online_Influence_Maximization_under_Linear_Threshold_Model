import networkx as nx
import random
import matplotlib.pyplot as plt
import pickle

def copy_G_random(G_origin):
    # Generate Watts-Strogatz (WS) Graph as a directed graph
    G = G_origin.copy()
    
    # Convert to directed graph
    #G = G.to_directed()

    # Add random weights to edges
    dir_of_edgeWeight = {}
    for edge in G.edges():
        weight = random.uniform(0, 1)
        dir_of_edgeWeight[edge] = weight
        # エッジが存在するかどうか
        G[edge[0]][edge[1]]['weight'] = 1

    # Normalize edge weights for each node's in-edges
    for node in G.nodes():
        sum_of_edge_weight_V = 0
        for edge in G.in_edges(node):  # Now it works since G is directed
            sum_of_edge_weight_V += dir_of_edgeWeight[edge]
        if sum_of_edge_weight_V > 1 + 1e-4:
            for edge in G.in_edges(node):
                normalized_weight = dir_of_edgeWeight[edge] / sum_of_edge_weight_V
                dir_of_edgeWeight[edge] = normalized_weight
                # Update the normalized weight in the graph as well
                G[edge[0]][edge[1]]['weight'] = 1

    EwTrue = dir_of_edgeWeight
    return G, EwTrue

def copy_G_edgenum(G_origin):
        # 元のグラフをコピー
    G = G_origin.copy()

    # エッジの重みを格納する辞書
    dir_of_edgeWeight = {}

    # 各ノードの入次数に基づいてエッジの重みを設定
    for node in G.nodes():
        in_edges = list(G.in_edges(node))  # ノードへの入エッジリスト
        num_in_edges = len(in_edges)
        if num_in_edges > 0:
            # 基準の重みを入次数分の1に設定
            base_weight = 1 / num_in_edges
            for edge in in_edges:
                # ランダムなノイズを加える（例: ノイズの範囲は±5%）
                noise = random.uniform(-0.1 * base_weight, 0.05 * base_weight)
                weight = base_weight + noise
                dir_of_edgeWeight[edge] = weight
                # グラフのエッジに重みを設定
                G[edge[0]][edge[1]]['weight'] = 1

    # 各ノードの入エッジの重みを正規化
    for node in G.nodes():
        sum_of_edge_weight_V = 0
        for edge in G.in_edges(node):
            sum_of_edge_weight_V += dir_of_edgeWeight[edge]
        if sum_of_edge_weight_V > 0:  # 重みが0でない場合のみ正規化
            for edge in G.in_edges(node):
                normalized_weight = dir_of_edgeWeight[edge] / sum_of_edge_weight_V
                dir_of_edgeWeight[edge] = normalized_weight
                # 正規化された重みをグラフに反映
                G[edge[0]][edge[1]]['weight'] = 1

    EwTrue = dir_of_edgeWeight
    return G, EwTrue
import os
import pickle
import networkx as nx
import matplotlib.pyplot as plt

# バイナリファイルのディレクトリパス
directory_path = '.'

# ディレクトリ内のすべての .G ファイルを取得
file_list = [f for f in os.listdir(directory_path) if f.endswith('.G')]

for file_name in file_list:
    file_path = os.path.join(directory_path, file_name)
    
    # バイナリファイルを読み込んでグラフを復元
    with open(file_path, 'rb') as f:
        graph = pickle.load(f)
    
    # グラフの基本情報を表示
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()
    print(f"Graph {file_name} with {num_nodes} nodes and {num_edges} edges")
    
    # ノードとエッジの詳細を表示
    print("Nodes:")
    for node, data in graph.nodes(data=True):
        print(f"{node}: {data}")
    
    print("Edges:")
    for u, v, data in graph.edges(data=True):
        print(f"{u} -> {v}: {data}")

    # グラフの可視化
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, with_labels=True, node_size=500, node_color="skyblue", font_size=10, font_weight="bold", edge_color="gray")
    
    # 可視化を PNG ファイルとして保存
    png_file_path = os.path.join(directory_path, file_name.replace('.G', '.png'))
    plt.savefig(png_file_path)
    plt.clf()  # 次のグラフの描画のためにクリア

print("ERグラフの可視化が完了しました。")

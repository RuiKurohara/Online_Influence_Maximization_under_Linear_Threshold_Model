import os
import ast
import networkx as nx
from pyvis.network import Network

def parse_imports(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        tree = ast.parse(file.read(), filename=filepath)
    
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)
    return imports

def build_dependency_graph(directory):
    graph = nx.DiGraph()
    python_files = [f for f in os.listdir(directory) if f.endswith('.py')]
    
    for file in python_files:
        file_path = os.path.join(directory, file)
        imports = parse_imports(file_path)
        for imp in imports:
            graph.add_edge(file, imp + '.py')
    
    return graph

def visualize_graph(graph, output_path):
    net = Network(directed=True, notebook=True)
    
    for node in graph.nodes:
        net.add_node(node, label=node)
    
    for edge in graph.edges:
        net.add_edge(edge[0], edge[1])
    
    net.show(output_path)

def main(directory, output_path):
    graph = build_dependency_graph(directory)
    visualize_graph(graph, output_path)

if __name__ == '__main__':
    directory = './'  # 依存関係を可視化したいディレクトリを指定
    output_path = 'dependency_graph.html'  # 出力ファイルのパスを指定
    main(directory, output_path)

import os
import ast
import pydot

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

def build_dependency_graph(directory, graph=None, parsed_files=None):
    if graph is None:
        graph = pydot.Dot(graph_type='digraph')
    if parsed_files is None:
        parsed_files = set()

    for root, dirs, files in os.walk(directory):
        python_files = [f for f in files if f.endswith('.py')]

        for file in python_files:
            file_path = os.path.join(root, file)
            if file_path in parsed_files:
                continue
            parsed_files.add(file_path)
            imports = parse_imports(file_path)
            if imports:
                print(f"File: {file_path} imports: {imports}")
            for imp in imports:
                imp_file = imp.replace('.', '/') + '.py'
                imp_file_path = os.path.join(directory, imp_file)
                if os.path.exists(imp_file_path):
                    edge = pydot.Edge(file_path, imp_file_path)
                    graph.add_edge(edge)
                    build_dependency_graph(os.path.dirname(imp_file_path), graph, parsed_files)
    
    return graph

def visualize_graph(graph, output_path):
    graph.write_png(output_path)

def main(directory, output_path):
    graph = build_dependency_graph(directory)
    visualize_graph(graph, output_path)

if __name__ == '__main__':
    directory = './'  # 依存関係を可視化したいディレクトリを指定
    output_path = 'dependency_graph.png'  # 出力ファイルのパスを指定
    main(directory, output_path)

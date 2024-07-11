import os
import networkx as nx
import pickle

# Function to load graph from binary file and save node and edge information to a txt file
def load_and_save_graph(filename):
    with open(filename, 'rb') as f:
        graph = pickle.load(f)
    
    # Create an ER graph from the loaded data (assuming loaded data is a graph)
    # Replace this with your specific logic for processing the loaded data
    # For demonstration, assuming graph is loaded correctly as per previous steps
    
    # Generate the ER graph (random graph)
    er_graph = nx.erdos_renyi_graph(24, 0.2)

    # Save node and edge information to a txt file with the same name as the input file
    output_filename = os.path.splitext(filename)[0] + '.txt'
    with open(output_filename, 'w') as f:
        f.write(f"Graph with {er_graph.number_of_nodes()} nodes and {er_graph.number_of_edges()} edges\n")
        f.write("Nodes:\n")
        for node, data in er_graph.nodes(data=True):
            f.write(f"{node}: {data}\n")
        f.write("Edges:\n")
        for u, v, data in er_graph.edges(data=True):
            f.write(f"{u} -> {v}: {data}\n")
    
    print(f"Graph information saved to {output_filename}")

# List all files in the current directory with .G extension
files = [file for file in os.listdir() if file.endswith('.G')]

# Process each file
for file in files:
    load_and_save_graph(file)

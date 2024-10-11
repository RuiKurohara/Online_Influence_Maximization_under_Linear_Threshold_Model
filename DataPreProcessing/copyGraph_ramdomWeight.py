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
        # Set the weight as an attribute in the graph
        G[edge[0]][edge[1]]['weight'] = weight

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
                G[edge[0]][edge[1]]['weight'] = normalized_weight

    EwTrue = dir_of_edgeWeight
    return G, EwTrue

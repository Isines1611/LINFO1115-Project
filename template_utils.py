# If needed, write here your additional fuctions/classes with their signature and use them in the exercices:
# a specific place is available to copy them at the end of the Inginious task.

# First, import the libraries needed for your helper functions
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt # For plots
import networkx as nx# Tests

# Q1 Helper

def node_degree(df, node):
    src_degree = len(df[df['Src'] == node])
    dest_degree = len(df[df['Dst'] == node])

    return src_degree + dest_degree

# Q1 / Q2 Helper

def common_neighbors(df, n1, n2, distinct=False):
    neighbors_node1 = set(df[df['Src'] == n1]['Dst']) | set(df[df['Dst'] == n1]['Src'])
    neighbors_node2 = set(df[df['Src'] == n2]['Dst']) | set(df[df['Dst'] == n2]['Src'])

    if(distinct == True):
        common_neighbors = neighbors_node1 | neighbors_node2
    else:
        common_neighbors = neighbors_node1 & neighbors_node2
    
    return len(common_neighbors)

# Q5 Helpers

def get_adj_list(df):
    adj_list = {}
    
    for index, row in df.iterrows():
        src = row['Src']
        dst = row['Dst']

        if src not in adj_list:
            adj_list[src] = []
        
        if dst not in adj_list:
            adj_list[dst] = []

        adj_list[src].append(dst)
        adj_list[dst].append(src)
    
    return adj_list

# Plot Diagrams Functions

def plot_degree_distribution(df): # Histogram
    all_nodes = pd.concat([df['Src'], df['Dst']])
    node_counts = all_nodes.value_counts()

    plt.hist(node_counts, bins=range(1, max(node_counts)+2), align='left', edgecolor='black')
    plt.xlabel('Degree')
    plt.ylabel('Number of Nodes')
    plt.title('Degree Distribution')
    plt.grid(axis='y', alpha=0.75)
    plt.savefig("q1.png")
    plt.show()

def plot_similartiy_distribution(scores):
    sorted_scores = np.sort(scores)
    
    n = len(sorted_scores)
    cumulative_percentage = np.arange(1, n + 1) / n * 100
    
    plt.plot(sorted_scores, cumulative_percentage, marker='o', linestyle='-')
    plt.xlabel('Similarity Score')
    plt.ylabel('Percentage of Edges (%)')
    plt.title('Cumulative Distribution of Similarity Scores')
    plt.grid(True)
    plt.savefig("q2.png")
    plt.show()

# Q1

def get_total_local_bridges(df):
    total = 0

    for index, row in df.iterrows():
        src = row['Src']
        dst = row['Dst']

        total_both = common_neighbors(df, src, dst)

        if(total_both) == 0:
            total += 1

    return total

def get_total_bridges(df): # pour le coup c'est pas moi qui l'ai fait
    graph = {}  # create an adjacency list representation of the graph
    for index, row in df.iterrows():
        src = row['Src']
        dest = row['Dst']

        if src in graph:
            graph[src].append(dest)
        else:
            graph[src] = [dest]

        if dest in graph:
            graph[dest].append(src)
        else:
            graph[dest] = [src]

    num_bridges = 0
    visited = set()
    parent = {}
    low = {}
    disc = {}

    def dfs(node):
        nonlocal num_bridges
        visited.add(node)
        disc[node] = low[node] = len(visited)
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                parent[neighbor] = node
                dfs(neighbor)
                low[node] = min(low[node], low[neighbor])
                if low[neighbor] > disc[node]:
                    num_bridges += 1
            elif neighbor != parent.get(node, None):
                low[node] = min(low[node], disc[neighbor])

    for node in graph:
        if node not in visited:
            dfs(node)

    return num_bridges


# Q2

def similarity_scores(df):
    scores = []

    for index, row in df.iterrows():
        src = row['Src']
        dst = row['Dst']

        neighbors_src = set(df[df['Src'] == src]['Dst']) | set(df[df['Dst'] == src]['Src'])
        neighbors_dst = set(df[df['Src'] == dst]['Dst']) | set(df[df['Dst'] == dst]['Src'])

        common_neighbors = len(neighbors_src.intersection(neighbors_dst))
        total_distinct_neighbors = len(neighbors_src.union(neighbors_dst))

        similarity_score = common_neighbors / total_distinct_neighbors
        scores.append(similarity_score)

    return scores

# Q3

def pagerank(df):
    damping_factor = 0.85
    convergence_threshold = 1e-6

    nodes = np.unique(np.concatenate((df['Src'].unique(), df['Dst'].unique())))

    node_index = {node: i for i, node in enumerate(nodes)}
    num_nodes = len(nodes)

    transition_matrix = np.zeros((num_nodes, num_nodes))

    outgoing_links_count = np.zeros(num_nodes)
    for src, dst in zip(df['Src'], df['Dst']):
        src_index = node_index[src]
        dst_index = node_index[dst]
        transition_matrix[src_index, dst_index] = 1
        outgoing_links_count[src_index] += 1

    # normalization
    nonzero_indices = np.where(outgoing_links_count > 0)[0]
    transition_matrix[:, nonzero_indices] /= outgoing_links_count[nonzero_indices][np.newaxis, :]

    scores = np.ones(num_nodes) / num_nodes

    while True: # Use max iteration if infinite loop
        new_scores = (1 - damping_factor) / num_nodes + damping_factor * transition_matrix.T.dot(scores)
        total_update = np.sum(np.abs(new_scores - scores))

        if total_update < convergence_threshold:
            break

        scores = new_scores

    scores = {node: score for node, score in zip(nodes, scores)}

    return scores

# Q5

def get_betweenness_centrality(adj): # en 2min
    betweenness_centrality = {node: 0 for node in adj.keys()}

    for source in adj.keys():
        visited = {source}
        queue = [(source, [source])]

        while queue:
            curr, path = queue.pop(0)

            for neighbor in adj[curr]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
                    
                    for nb in (path+[neighbor])[1:-1]:
                        betweenness_centrality[nb] += 1

            

    return betweenness_centrality




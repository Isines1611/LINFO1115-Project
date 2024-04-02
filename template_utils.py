# If needed, write here your additional fuctions/classes with their signature and use them in the exercices:
# a specific place is available to copy them at the end of the Inginious task.

# First, import the libraries needed for your helper functions
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_all_nodes(df):
    nodes = []
    for index, row in df.iterrows():
        src = row['Src']
        dst = row['Dst']

        if src not in nodes:
            nodes.append(src)
        if dst not in nodes:
            nodes.append(dst)

    return nodes


# Q1 Helpers

def node_degree(df, node):
    src_degree = len(df[df['Src'] == node])
    dest_degree = len(df[df['Dst'] == node])

    return src_degree + dest_degree

def common_neighbors(df, n1, n2, distinct=False):
    neighbors_node1 = set(df[df['Src'] == n1]['Dst']) | set(df[df['Dst'] == n1]['Src'])
    neighbors_node2 = set(df[df['Src'] == n2]['Dst']) | set(df[df['Dst'] == n2]['Src'])

    if(distinct == True):
        common_neighbors = neighbors_node1 | neighbors_node2
    else:
        common_neighbors = neighbors_node1 & neighbors_node2
    
    return len(common_neighbors)


# Q1

def get_total_local_bridges(df):
    total = 0

    for index, row in df.iterrows():
        src = row['Src']
        dst = row['Dst']

        total_both = common_neighbors(df, src, dst)
        total_neighbor = node_degree(df, src) + node_degree(df, dst) - 2

        print("\nFor",src,"to",dst,"\nBoth:",total_both,"total:",total_both,"no=", (total_both / total_neighbor),"\n")

        if(total_both / total_neighbor) == 0:
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

def plot_degree_distribution(df): # Histogram
    all_nodes = pd.concat([df['Src'], df['Dst']])
    
    node_counts = all_nodes.value_counts()

    # Plot histogram of degree distribution
    plt.hist(node_counts, bins=range(1, max(node_counts)+2), align='left', edgecolor='black')
    plt.xlabel('Degree')
    plt.ylabel('Number of Nodes')
    plt.title('Degree Distribution')
    plt.grid(axis='y', alpha=0.75)
    plt.show()

# Q2
def similarity_scores(df):
    scores = []

    for index, row in df.iterrows():
        src = row['Src']
        dst = row['Dst']

        same_neighbors = common_neighbors(df, src, dst)
        distinct_neighbors = common_neighbors(df, src, dst, True)

        scores.append(same_neighbors/distinct_neighbors)

    return scores

def plot_similartiy_distribution(scores):
    plt.hist(scores, bins=20, edgecolor='black')
    plt.xlabel('Similarity Score')
    plt.ylabel('Frequency')
    plt.title('Similarity Score Distribution')
    plt.grid(axis='y', alpha=0.75)
    plt.show()
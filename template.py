import pandas as pd
import numpy as np
import sys 
from template_utils import *

sys.setrecursionlimit(6000)

# Undirected graph
# Task 1: Average degree, number of bridges, number of local bridges
def Q1(dataframe):

    total_degree = 0
    num_nodes = len(set(dataframe['Src']) | set(dataframe['Dst']))

    for node in set(dataframe['Src']) | set(dataframe['Dst']):
        total_degree += node_degree(dataframe, node)

    average_degree = total_degree / num_nodes if num_nodes > 0 else 0

    #plot_degree_distribution(dataframe) # Plot the histogram

    bridges = get_total_bridges(dataframe) # Bridges
    local_bridges = get_total_local_bridges(dataframe) # Local bridges

    return [average_degree, bridges, local_bridges] # [average degree, nb bridges, nb local bridges]

# Undirected graph
# Task 2: Average similarity score between neighbors
def Q2(dataframe):

    scores = similarity_scores(dataframe)

    plot_similartiy_distribution(scores) # plot the histogram

    return np.mean(scores) # the average similarity score between neighbors

# Directed graph
# Task 3: PageRank
def Q3(dataframe):

    scores = pagerank(dataframe)
    index = max(scores, key=scores.get)

    return [index, scores[index]] # the id of the node with the highest pagerank score, the associated pagerank value.
    # Note that we consider that we reached convergence when the sum of the updates on all nodes after one iteration of PageRank is smaller than 10^(-6)

# Undirected graph
# Task 4: Small-world phenomenon
def Q4(dataframe):
    dist,length = Floyd_Warshall(dataframe)
    dist = dist.tolist()
    length = length.tolist()
    number_of_len = np.zeros(4950).tolist()
    for i in range(len(length)):
        for j in range(len(length)) :
            number_of_len[int(length[i][j])]+=1
    diameter = -float('inf')
    for i in range(len(dist)):
        for j in range(len(dist)):
            if(diameter < dist[i][j]) : 
                diameter = dist[i][j]
                src_node = i+1
                dest_node = j+1
    most_common_length = np.argmax(number_of_len)
    plot_number_path_for_length(number_of_len)
    return [src_node,dest_node,diameter,most_common_length]# at index 0 the number of shortest paths of lenght 0, at index 1 the number of shortest paths of length 1, ...
    # Note that we will ignore the value at index 0 as it can be set to 0 or the number of nodes in the graph

# Undirected graph
# Task 5: Betweenness centrality
def Q5(dataframe):

    adj_list = get_adj_list(dataframe)
    betweenness_centrality = get_betweenness_centrality(adj_list)

    index = max(betweenness_centrality, key=betweenness_centrality.get)

    return [index, betweenness_centrality[index]] # the id of the node with the highest betweenness centrality, the associated betweenness centrality value.

# you can write additionnal functions that can be used in Q1-Q5 functions in the file "template_utils.py", a specific place is available to copy them at the end of the Inginious task.

df = pd.read_csv('powergrid.csv')
#df = pd.read_csv('new.csv')
#print("Q1", Q1(df))
#print("Q2", Q2(df))
#print("Q3", Q3(df))
print("Q4", Q4(df))
#print("Q5", Q5(df))
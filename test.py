import networkx as nx
import pandas as pd
import numpy as np

df = pd.read_csv('powergrid.csv')
'''
G = nx.from_pandas_edgelist(df, source='Src', target='Dst')

pagerank_score = nx.pagerank(G)
for i in pagerank_score.keys() :
    print(pagerank_score[i])
highest_pagerank_score_node = max(pagerank_score, key=pagerank_score.get)
highest_pagerank_score_value = pagerank_score[highest_pagerank_score_node]

print(highest_pagerank_score_node, highest_pagerank_score_value)'''
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
scores = pagerank(df)
index = max(scores, key=scores.get)

print(index, scores[index])
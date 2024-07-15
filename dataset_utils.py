import networkx as nx
import itertools
import pandas as pd
import numpy as np
import random
import pickle
import os

from TemporalGraph import TemporalGraph

DEFAULT_FEATURE_METRICS = [nx.jaccard_coefficient, nx.adamic_adar_index, nx.resource_allocation_index, nx.preferential_attachment]

def edges_in_graph(g: nx.Graph, edges_subset):
    return [1 if g.has_edge(edge[0], edge[1]) else 0 for edge in edges_subset]

def generate_features(g: nx.Graph, edges_subset, feature_metrics=DEFAULT_FEATURE_METRICS):    
    metrics_df = pd.DataFrame(index=pd.Index(edges_subset))
    for feature_metric in feature_metrics:
        metrics_df[feature_metric.__name__] = [_tuple[2] for _tuple in list(feature_metric(g, edges_subset))]
    return metrics_df


def generate_temporal_features(tg: TemporalGraph, edges_subset, times_subset, feature_metrics=DEFAULT_FEATURE_METRICS):
    features = []
    for time in times_subset:
        features.append(generate_features(tg.get_graph_at_time(time), edges_subset, feature_metrics))
    df = pd.concat(features, axis=1)
    df.columns = pd.MultiIndex.from_product([times_subset, [feature_metric.__name__ for feature_metric in feature_metrics]])
    return df

def extract_balanced_dataset(g: nx.Graph, candidate_negative_edges=[], verbose=False):
    positive_edges = [tuple(sorted(edge)) for edge in list(g.edges())]
    tmp_candidate_negative_edges = candidate_negative_edges.copy()
    random.shuffle(tmp_candidate_negative_edges)

    n_positive_edges = len(positive_edges)
    n_candidate_negative_edges = len(tmp_candidate_negative_edges)
    n_random_negative_edges = max(n_positive_edges - n_candidate_negative_edges, 0)

    if verbose:
        print(f"\t(positive edges/candidate negative edges): ({n_positive_edges}/{n_candidate_negative_edges})")
        print(f"\tRandom negative edges: {n_random_negative_edges} ({n_random_negative_edges/(n_random_negative_edges + n_candidate_negative_edges)*100}%)")
        print()

    all_possible_edges, negative_balanced_edges = list(itertools.combinations(g.nodes, 2)), []
    while len(negative_balanced_edges) < n_positive_edges:
        negative_candidate = tmp_candidate_negative_edges.pop() if tmp_candidate_negative_edges else random.choice(all_possible_edges)
        if not negative_candidate in positive_edges:
            negative_balanced_edges.append(negative_candidate)
    return positive_edges + negative_balanced_edges


def generate_snapshot_features_labels(tg: TemporalGraph, snapshot_time, n_past_snapshot_to_consider, feature_metrics=DEFAULT_FEATURE_METRICS, candidate_negative_edges=[], verbose=False):
    training_graph = tg.get_graph_at_time(snapshot_time)
    if verbose:
        print(f"Dataset statistics for snapshot {snapshot_time}:")
    edges = extract_balanced_dataset(training_graph, candidate_negative_edges, verbose)
    training_timestamps = list(range(snapshot_time-n_past_snapshot_to_consider, snapshot_time))

    features = np.array(generate_temporal_features(tg, edges, training_timestamps, feature_metrics))
    labels = np.array(edges_in_graph(training_graph, edges))

    return edges, features, labels

# TODO precompute and store features in a feature database for performance
# negative_sampling can be ["random", "historical_negative", "inductive_negative"]
def generate_dataset(tg: TemporalGraph, n_past_snapshot_to_consider, feature_metrics=DEFAULT_FEATURE_METRICS, negative_sampling="random", verbose=False):
    edges, features, labels = [], [], []

    for snapshot_time in range(n_past_snapshot_to_consider, tg.max_time() + 1): 
        if negative_sampling == "random":
            candidate_negative_edges = []
        elif negative_sampling == "historical_negative":
            candidate_negative_edges = tg.get_positive_edges_up_to_snapshot(snapshot_time - 1)
        elif negative_sampling == "inductive_negative":
            candidate_negative_edges = tg.get_positive_edges_from_snapshot(snapshot_time - 1)
        else:
            raise Exception("Sampling strategy not implemented!")
        snapshot_edges, snapshot_features, snapshot_labels = \
        generate_snapshot_features_labels(tg, snapshot_time, n_past_snapshot_to_consider, feature_metrics, candidate_negative_edges, verbose)

        edges.append(snapshot_edges)
        features.append(snapshot_features)
        labels.append(snapshot_labels)

    return edges, features, labels

def load_or_generate_dataset(tg: TemporalGraph, n_past_snapshot_to_consider, feature_metrics=DEFAULT_FEATURE_METRICS, negative_sampling="random", verbose=False):
    datasets_common_path = f'datasets/{n_past_snapshot_to_consider}_past_snapshot/{negative_sampling}'

    # Assuming that if edges exists also features/labels do
    if os.path.exists(f"{datasets_common_path}/edges.pkl"):
        edges = pickle.load(open(f'{datasets_common_path}/edges.pkl', 'rb'))
        features = pickle.load(open(f'{datasets_common_path}/features.pkl', 'rb'))
        labels = pickle.load(open(f'{datasets_common_path}/labels.pkl', 'rb'))
    else:
        edges, features, labels = generate_dataset(tg, n_past_snapshot_to_consider, feature_metrics, negative_sampling, verbose)
        save_dataset(datasets_common_path, edges, features, labels)
    
    return edges, features, labels


def save_dataset(datasets_common_path, edges, features, labels):
    os.makedirs(datasets_common_path)
    pickle.dump(edges, open(f'{datasets_common_path}/edges.pkl', 'wb'))
    pickle.dump(features, open(f'{datasets_common_path}/features.pkl', 'wb'))
    pickle.dump(labels, open(f'{datasets_common_path}/labels.pkl', 'wb'))


def basic_analysis(graph):
    print("Number of nodes:", graph.number_of_nodes())
    print("Number of edges:", graph.number_of_edges())
    print("Graph density:", nx.density(graph))
    print("Average shortest path length:", nx.average_shortest_path_length(graph))
    print("Connected components:", list(nx.connected_components(graph)))
    print("Degree centrality:", nx.degree_centrality(graph))
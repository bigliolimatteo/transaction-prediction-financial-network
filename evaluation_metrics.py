import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from dataset_utils import generate_snapshot_features_labels, extract_balanced_dataset, generate_temporal_features, edges_in_graph, DEFAULT_FEATURE_METRICS
import itertools
import random
import math
import networkx as nx
from itertools import compress
import torch 

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from catboost import CatBoostClassifier
import lightgbm as lgb
from t3gnn import T3GNN

def flatten(l):
    return [item for sublist in l for item in sublist]

def evaluation_metrics():
    return {
        "accuracy_score": accuracy_score,
        "precision_score": precision_score,
        "recall_score": recall_score,
        "f1_score": f1_score,
        "roc_auc_score": roc_auc_score,
        "average_precision_score": average_precision_score
    }

def compute_evaluation_metrics(test_values, predicted_values):
    return {key: value(test_values, predicted_values) for key, value in evaluation_metrics().items()}

def evaluate_model_on_test_data(model, test_features, test_values, index=0):
    if isinstance(model, T3GNN): 
        test_values = test_features.edge_label.cpu().detach().numpy()
        predicted_values = np.rint(roland_prediction(model, test_features, index))
        return compute_evaluation_metrics(test_values, predicted_values)
    return compute_evaluation_metrics(test_values, model.predict(test_features))

def evaluate_edgebank_baseline(train_edges, test_edges, train_labels, test_labels):

    positive_edges_list = list()
    for (train_edges_t, train_labels_t) in zip(train_edges, train_labels):
        positive_edges_mask = np.array(train_labels_t).astype(bool)
        positive_edges_list.append(compress(train_edges_t, positive_edges_mask))
    already_seen_edges = set(flatten(positive_edges_list))

    predicted_values = []
    for test_edges_t in test_edges:
        predicted_values.append(np.array([edge in already_seen_edges for edge in test_edges_t]).astype(int))
        already_seen_edges.update(set(test_edges_t))

    return compute_evaluation_metrics(np.concatenate(test_labels), np.concatenate(predicted_values))


def evaluate_models_on_test_data(models_list, test_features, test_values, edgebank_baseline_data = None):
    # Assume that models list is a list of tuples like
    # [(model_name_a, model_a), (model_name_b, model_b), ...]

    models_metrics = [evaluate_model_on_test_data(model, test_features, test_values) for _, model in models_list]
    models_names = [model_name for model_name, _ in models_list]

    if edgebank_baseline_data is not None:
        train_edges = edgebank_baseline_data["train_edges"]
        test_edges = edgebank_baseline_data["test_edges"]
        train_labels = edgebank_baseline_data["train_labels"]
        test_labels = edgebank_baseline_data["test_labels"]
        n_past_snapshots = edgebank_baseline_data["n_past_snapshots"]
        
        models_metrics = models_metrics + [evaluate_edgebank_baseline(train_edges, test_edges, train_labels, test_labels)]
        models_names = models_names + [f"edgebank_baseline_{n_past_snapshots}_past_snapshots"]

    return pd.DataFrame(models_metrics, index=models_names).T

def roland_prediction(model, test_data, isnap, device='cpu'):
    model.eval()
    test_data = test_data.to(device)
    h, _ = model(test_data.x, test_data.edge_index, edge_label_index = test_data.edge_label_index, isnap=isnap)
    return torch.sigmoid(h).cpu().detach().numpy()


def roland_validation(model, test_data, isnap, device='cpu'):
    pred_cont_link = roland_prediction(model, test_data, isnap, device)
    label_link = test_data.edge_label.cpu().detach().numpy()
    avgpr_score_link = average_precision_score(label_link, pred_cont_link)
    return avgpr_score_link

from sklearn.metrics import *

def roland_train_single_snapshot(model, data, train_data, val_data, isnap,\
                          last_embeddings, optimizer, device='cpu', num_epochs=50, verbose=False):
    
    avgpr_val_max = 0
    best_model = model
    train_data = train_data.to(device)
    best_epoch = -1
    best_current_embeddings = []
    
    avgpr_trains = []
    #avgpr_vals = []
    avgpr_tests = []
    
    tol = 1
    
    for epoch in range(num_epochs):
        model.train()
        ## Note
        ## 1. Zero grad the optimizer
        ## 2. Compute loss and backpropagate
        ## 3. Update the model parameters
        optimizer.zero_grad()

        pred,\
        current_embeddings =\
            model(train_data.x, train_data.edge_index, edge_label_index = train_data.edge_label_index,\
                  isnap=isnap, previous_embeddings=last_embeddings)
        
        loss = model.loss(pred, train_data.edge_label.type_as(pred)) #loss to fine tune on current snapshot

        loss.backward(retain_graph=True)  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.

        ##########################################

        log = 'Epoch: {:03d}\n AVGPR Train: {:.4f}, Val: {:.4f}, Test: {:.4f}\n MRR Train: {:.4f}, Val: {:.4f}, Test: {:.4f}\n F1-Score Train: {:.4f}, Val: {:.4f}, Test: {:.4f}\n Loss: {}'
        avgpr_score_val  = roland_validation(model, val_data, isnap, device)
        
        if avgpr_val_max-tol <= avgpr_score_val:
            avgpr_val_max = avgpr_score_val
            best_epoch = epoch
            best_current_embeddings = current_embeddings
            best_model = model
        else:
            break
                    
    return best_model, optimizer, best_current_embeddings



def evaluate_model_in_live_update_setting(g, model, features, labels, n_past_snapshot_to_consider, max_time):

    def prepare_train_test(index, snapshot_time, training_set='complete'):
        index_to_predict = index + 1
        snapshot_time_to_predict = snapshot_time + 1

        if training_set == 'complete':
            train_features, train_labels = np.concatenate(features[:index_to_predict]), np.concatenate(labels[:index_to_predict])
        elif training_set == 'last':
            train_features, train_labels = features[index_to_predict-1], labels[index_to_predict-1]
        else:
            raise Exception("Training set not defined")
        test_features, test_labels = features[index_to_predict], labels[index_to_predict]

        return snapshot_time_to_predict, train_features, train_labels, test_features, test_labels

    def torch_snapshot_from_graph(g, snapshot_time):
        from torch_geometric.data import Data
        from torch_geometric.transforms import Constant

        snapshot = Data()
        snapshot.num_nodes = len(g.unique_nodes())
        snapshot.edge_index = torch.tensor(np.array(g.get_graph_at_time(snapshot_time).edges()).transpose())
        constant = Constant()
        return constant(snapshot) 

    def prepare_t3gnn_data(g, snapshot_time):
        from torch_geometric.transforms import RandomLinkSplit

        train_snapshot = torch_snapshot_from_graph(g, snapshot_time)
        transform = RandomLinkSplit(num_val=0.0,num_test=0.25)
        train_data, _, val_data = transform(train_snapshot)

        test_data = torch_snapshot_from_graph(g, snapshot_time + 1)

        #NEGATIVE SET: EDGES CLOSED IN THE PAST BUT NON IN THE CURRENT TEST SET
        past_edges = set(zip([int(e) for e in train_snapshot.edge_index[0]],\
                             [int(e) for e in train_snapshot.edge_index[1]]))
        current_edges = set(zip([int(e) for e in test_data.edge_index[0]],\
                             [int(e) for e in test_data.edge_index[1]]))
        
        negative_edges = list(past_edges.difference(current_edges))[:test_data.edge_index.size(1)]
        future_neg_edge_index = torch.Tensor([[a[0] for a in negative_edges],\
                                                 [a[1] for a in negative_edges]]).long()
        
        num_pos_edge = test_data.edge_index.size(1)
        num_neg_edge = future_neg_edge_index.size(1)
        test_data.edge_label = torch.Tensor(np.array([1 for i in range(num_pos_edge)] + [0 for i in range(num_neg_edge)]))
        test_data.edge_label_index = torch.cat([test_data.edge_index, future_neg_edge_index], dim=-1)

        return snapshot_time + 1, train_snapshot, train_data, val_data, test_data

    snapshot_times, metrics = [], []
    if isinstance(model, T3GNN):
        _, snapshot_0, _, _, _ = prepare_t3gnn_data(g, 0)
        num_nodes = snapshot_0.x.size(0)
        last_embeddings = [torch.Tensor([[0 for i in range(model.hidden_dim)] for j in range(num_nodes)])]
        rolopt = torch.optim.Adam(params=model.parameters(), lr=0.01, weight_decay = 5e-3)

    for index, snapshot_time in enumerate(range(n_past_snapshot_to_consider-1, max_time-1)):

        if isinstance(model, KNeighborsClassifier):
            snapshot_time_to_predict, train_features, train_labels, test_features, test_labels = prepare_train_test(index, snapshot_time, 'complete')
            model.fit(train_features, train_labels)
        
        if isinstance(model, MLPClassifier):
            snapshot_time_to_predict, train_features, train_labels, test_features, test_labels = prepare_train_test(index, snapshot_time, 'last')
            model.partial_fit(train_features, train_labels)
        
        # Missing random forest

        if isinstance(model, CatBoostClassifier):
            snapshot_time_to_predict, train_features, train_labels, test_features, test_labels = prepare_train_test(index, snapshot_time, 'last')
            init_model_path = "tmp_models/catboost.cbm" if index > 0 else None
            model.fit(train_features, train_labels, init_model = init_model_path)
            model.save_model("tmp_models/catboost.cbm")

        if isinstance(model, lgb.LGBMClassifier):
            snapshot_time_to_predict, train_features, train_labels, test_features, test_labels = prepare_train_test(index, snapshot_time, 'last')
            init_model_path = "tmp_models/lightgbm.cbm" if index > 0 else None
            model.fit(train_features, train_labels, init_model = init_model_path)
            model.booster_.save_model("tmp_models/lightgbm.cbm")
        
        if isinstance(model, T3GNN):
            snapshot_time_to_predict, snapshot, train_data, val_data, test_features = prepare_t3gnn_data(g, snapshot_time)
            test_labels = None
            model, rolopt, last_embeddings = \
                roland_train_single_snapshot(model, snapshot, train_data, val_data, snapshot_time, last_embeddings, rolopt)


        snapshot_times.append(snapshot_time_to_predict)
        metrics.append(evaluate_model_on_test_data(model, test_features, test_labels, index))
    
    return pd.DataFrame(metrics, index=snapshot_times).rename_axis('snapshot_time')

def evaluate_edgebank_in_live_update_setting(edges, labels, n_past_snapshot_to_consider, max_time):
    snapshot_times, metrics = [], []
    for index, snapshot_time in enumerate(range(n_past_snapshot_to_consider, max_time-1)):

        index_to_predict = index + 1
        snapshot_time_to_predict = snapshot_time + 1

        train_edges = edges[:index_to_predict]
        train_labels = labels[:index_to_predict]
        
        test_edges = [edges[index_to_predict]]
        test_labels = [labels[index_to_predict]]

        snapshot_times.append(snapshot_time_to_predict)
        metrics.append(evaluate_edgebank_baseline(train_edges, test_edges, train_labels, test_labels))
    
    return pd.DataFrame(metrics, index=snapshot_times).rename_axis('snapshot_time')

def compare_score_across_models(score_list, metric_chosen = "average_precision_score"):
    # Assume that score list is a list of scores_df like
    # [(model_name_a, scores_df_a), (model_name_b, scores_df_b), ...]

    metrics_df = pd.DataFrame()
    metrics_df.index = score_list[0][1].index

    for model_name, score_df in score_list:
        metrics_df[f"{model_name} - {metric_chosen}"] = score_df[metric_chosen]
    metrics_df.plot(figsize=(15, 5), grid=True, xticks=metrics_df.index)
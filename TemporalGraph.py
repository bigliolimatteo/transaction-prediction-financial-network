from dataclasses import dataclass
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class GraphWTime:
    time: int
    graph: nx.Graph

    def __iter__(self):
        return iter((self.time, self.graph))

    def __lt__(self, other):
         return self.time < other.time

class TemporalGraph:
    def __init__(self, graph_list: GraphWTime):
        self.graph_list = self._is_valid_graph_list(graph_list)

    def check_graph_list_time_uniqueness(self, graph_list: GraphWTime):
        time_list = [graph_w_time.time for graph_w_time in graph_list]
        return time_list, len(set(time_list)) == len(time_list)
    
    def _is_valid_graph_list(self, graph_list: GraphWTime):
        time_list, is_unique = self.check_graph_list_time_uniqueness(graph_list)
        if is_unique:
            return sorted(graph_list)
        raise ValueError(f"Times in graph_list must be unique, you are providing {sorted(time_list)}")    

    def time_list(self): return [graph_w_time.time for graph_w_time in self.graph_list]

    def max_time(self): return self.time_list()[-1]
    
    def get_graph_at_time(self, time: int):
        return dict([(g.time, g.graph) for g in self.graph_list])[time]
    
    def get_initial_graph(self):
        return self.get_graph_at_time(0)

    def unique_nodes(self):
        return list(set([node for _, graph in self.graph_list for node in graph.nodes()]))

    def unique_edges(self):
        return list(set(map(tuple, map(sorted, [edge for (time, graph) in self.graph_list for edge in graph.edges()]))))
    
    def get_positive_edges_from_snapshot(self, time: int):
        return list({edge for t in self.time_list()[time:] for edge in self.get_graph_at_time(t).edges()})
    
    def get_positive_edges_up_to_snapshot(self, time: int):
        return list({edge for t in self.time_list()[:time] for edge in self.get_graph_at_time(t).edges()})

    def basic_analisys(self):

        # Basic info
        print(f"Total (unique) nodes: {len(self.unique_nodes())}")
        print(f"Total (unique) edges: {len(self.unique_edges())}")

        # Data to Analyse
        time_list_to_analyse = self.time_list()[1:]
        graph_list_to_analyse = self.graph_list[1:]

        # Data Analysis
        number_of_edges = [len(g.edges) for _, g in graph_list_to_analyse]
        number_nodes_w_degree_gt_0 = [np.sum(np.array([degree for _, degree in list(snapshot.graph.degree)]) > 0) for snapshot in graph_list_to_analyse]
        density = [nx.density(snapshot.graph) for snapshot in graph_list_to_analyse]
        degrees_df = pd.concat([pd.DataFrame([(f"{snapshot.time}-{node}", degree) for node, degree in snapshot.graph.degree if degree>0]).set_index(0) for snapshot in graph_list_to_analyse])


        # N edges and N nodes plot
        fig, ax1 = plt.subplots(figsize=(10, 4))
        ax1.plot(time_list_to_analyse, number_of_edges, label="Number of Edges", marker='o', color="blue")
        ax1.set_ylabel("Number of Edges")
        ax1.tick_params(axis='y', labelcolor='blue')
        ax2 = ax1.twinx()
        ax2.plot(time_list_to_analyse, number_nodes_w_degree_gt_0, label="Number of Nodes w Degree > 0", marker='o', color="red")
        ax2.set_ylabel("Number of Nodes w Degree > 0")
        ax2.tick_params(axis='y', labelcolor='red')
        fig.legend()
        plt.show()

        # Degrees boxplot
        degrees_df.groupby(lambda x: x.split("-")[0]).boxplot(subplots=False, figsize=(20,4), rot=90, sym='')
        plt.show()

        fig, ax1 = plt.subplots(figsize=(10, 4))
        ax1.plot(time_list_to_analyse, density, label="Density", marker='o', color="blue")
        ax1.set_ylabel("Density")
        ax1.tick_params(axis='y', labelcolor='blue')
    
    def tea_plot(self, time_list = None):

        if time_list is None: time_list = self.time_list()

        new_edges = []
        repeated_edges = []
        already_observed_edges = set()

        for t in time_list:
            edges_t = set(self.get_graph_at_time(t).edges())
            n_edges_t = len(edges_t)
            n_already_observed_edges_t = len(already_observed_edges.intersection(edges_t))
            new_edges.append(n_already_observed_edges_t)
            repeated_edges.append(n_edges_t - n_already_observed_edges_t)
            already_observed_edges.update(edges_t)

        novelty = 1/len(time_list)*sum([new_edges_t/(new_edges_t + repeated_edges_t) 
                                        for (new_edges_t, repeated_edges_t) in zip(new_edges, repeated_edges)])
        
        print(f"Novelty metrics: {novelty}")
        
        metric_df = pd.DataFrame(list(map(list, zip(time_list, repeated_edges, new_edges))),
                  columns=['Time', 'Repeated Edges', 'New Edges'])
        
        metric_df.plot(x='Time', kind='bar', stacked=True, title='Tea Plot')
        plt.show()

    def reoccurence_and_surprise(self, time_list = None):

        if time_list is None: time_list = self.time_list()

        reoccurence_indices = []
        surprise_indices = []
        for t in time_list:
            e_train_t = self.get_positive_edges_up_to_snapshot(t-1)
            e_test_t = self.get_positive_edges_from_snapshot(t-1)

            reoccurence_indices.append(round(len(set(e_test_t).intersection(set(e_train_t)))/len(e_train_t), 3))
            surprise_indices.append(round(len(set(e_test_t).difference(set(e_train_t)))/len(e_test_t), 3))

        metric_df = pd.DataFrame(list(map(list, zip(reoccurence_indices, surprise_indices))),
                  columns=['Reoccurence Index', 'Surprise Index'], index=time_list)
        return metric_df
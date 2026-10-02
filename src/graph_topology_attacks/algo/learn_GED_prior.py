#############################################################
#### Import
#############################################################
import os
import joblib
import random
import pandas as pd
import numpy as np
import seaborn as sns
import networkx as nx
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline

from graph_topology_attacks.metrics import matrix_norm, Gram_matrix_norm
from graph_topology_attacks.utils import signal_propagation, setting_attacks


#############################################################
#### Generate data for regression
#############################################################
def reward_function(X_history, X_guess, metric='Gram_error_fro'):
    records = {}
    match metric:
        case 'fro':
            records = {metric: {t: [] for t in range(len(X_history))}}
            records = matrix_norm(X_history, X_guess, records, norm_type='fro')
        case 'Gram_error_fro':
            Gram_matrix_norm(X_history, X_guess, records, norm_type='fro')
        case _:
            raise ValueError(
                f"Unsupported reward metric '{metric}'. "
                "Supported metrics: 'fro', 'Gram_error_fro'."
            )

    error_o_time = list(records[metric].values())
    
    return error_o_time

def get_legal_moves(curr_node:int, G:nx.Graph):
    
    attackers_list = [node for node in G.nodes() if
                        str(node).startswith("attacker")]
    connected_nodes = set.union(*[nx.node_connected_component(G, attacker) for attacker in attackers_list])
    
    if curr_node in connected_nodes:
        possible_actions = [node for node in G.nodes() if
                node != curr_node and
                not str(node).startswith("attacker") and
                not G.has_edge(curr_node, node)]
        return possible_actions
    
    possible_actions = [n for n in connected_nodes if
                        n != curr_node and
                        not str(n).startswith("attacker") and
                        not G.has_edge(curr_node, n)]

    return possible_actions

def MonteCarlo_simulation(curr_node:int,G_t:nx.Graph, n_edges:int):
    """ Perform a random simulation untill reach the target number of edges.
    Return the last action to take and the second last graph state to allow a
    call to the play function to get the reward"""
    G_bis = G_t.copy()
    budget = n_edges - G_bis.number_of_edges()
    c_node = curr_node

    while budget > 0:
        moves = get_legal_moves(c_node, G_bis)
        if not moves:
            # Si le noeud de départ est déjà saturé, on en prend un autre !
            valid_nodes = [n for n in G_bis.nodes() if 
                           not str(n).startswith("attacker") and 
                           get_legal_moves(n, G_bis)]
            if not valid_nodes:
                return None
            c_node = random.choice(valid_nodes)
            moves = get_legal_moves(c_node, G_bis)
            
        action = random.choice(moves)
        if budget == 1:
            return c_node, action, G_bis
        
        G_bis.add_edge(c_node, action)
        budget -= 1
        c_node = random.choice([node for node in G_bis.nodes() if 
                                not str(node).startswith("attacker") and 
                                get_legal_moves(node, G_bis)])

    return None

def error_by_GED(target_graph, reward_label):
    nb_nodes = target_graph.number_of_nodes()
    time_step = 2 * nb_nodes

    error_by_r_edges = {"reward_label": [], "GED": [],
                        "nb_components": [], "nb_edges": [],
                        "nb_neighbors_attackers": [],
                        **{f"error_{t}": [] for t in range(time_step)}}

    total_edges = target_graph.number_of_edges()
    subset_nodes = random.sample(list(target_graph.nodes()), 1)

    for node in subset_nodes:
        a_mapping = {node: "attacker_0"}
        G_infected, attackers_neighbors = setting_attacks(target_graph, a_mapping)
        non_infected_node = [n for n in G_infected.nodes() if not str(n).startswith("attacker")]
        N_A = sum(len(neigh) for neigh in attackers_neighbors.values())
        attackers_set = list(attackers_neighbors.keys())
        X_history = signal_propagation(G_infected, attackers_set, N_A, time_step)
        # Generate data solution (GED=0)
        error = reward_function(X_history, X_history, metric=reward_label)
        error_by_r_edges["reward_label"].append(reward_label)
        error_by_r_edges["GED"].append(0)
        error_by_r_edges["nb_components"].append(nx.number_connected_components(G_infected))
        error_by_r_edges["nb_edges"].append(total_edges)
        error_by_r_edges["nb_neighbors_attackers"].append(N_A)

        for t, err in enumerate(error[:time_step]):
            error_by_r_edges[f"error_{t}"].append(err[0])

        possible_edges_to_move = [e for e in G_infected.edges() if 
                                  not (e[0] in attackers_set or e[1] in attackers_set)]

        for nb_edges_to_move in range(1, min(G_infected.number_of_edges(), G_infected.number_of_nodes() // 2 )):

            combs = [random.sample(possible_edges_to_move, nb_edges_to_move) for _ in range(5)]

            for edges_to_move in combs:

                modified_graph = G_infected.copy()
                modified_graph.remove_edges_from(edges_to_move)
                res = MonteCarlo_simulation(curr_node=random.choice(non_infected_node), G_t=modified_graph, n_edges=total_edges)

                if res is None:
                    continue

                c_node, action, modified_graph = res
                modified_graph.add_edge(c_node, action)

                X_guess = signal_propagation(modified_graph, attackers_set, N_A, T=time_step)
                error = reward_function(X_history, X_guess, metric=reward_label)

                error_by_r_edges["reward_label"].append(reward_label)
                error_by_r_edges["GED"].append(nx.graph_edit_distance(G_infected, modified_graph, timeout=7))
                # error_by_r_edges["GED"].append(nb_edges_to_move)
                error_by_r_edges["nb_components"].append(nx.number_connected_components(modified_graph))
                error_by_r_edges["nb_edges"].append(total_edges)
                error_by_r_edges["nb_neighbors_attackers"].append(N_A)

                for t, err in enumerate(error[:time_step]):
                    error_by_r_edges[f"error_{t}"].append(err[0])

    return error_by_r_edges

def export_data_e_by_GED(error_data:dict, data_name:str, graph_type:str):
    path_data = os.path.join("src", "graph_topology_attacks", "data", "models_data", graph_type)
    os.makedirs(path_data, exist_ok=True)

    file_path = os.path.join(path_data, data_name)
    df = pd.DataFrame(error_data)
    df.to_csv(file_path, mode="a", header=not os.path.isfile(file_path), index=False)


def split_data(df: pd.DataFrame):
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def generate_single_sample(args):
    i, n, graph_type = args

    if graph_type == "ER":
        p_bonus = np.linspace(0, 0.6, 10)
        p = 0.23 + p_bonus[i // 30]
        target_graph = nx.gnp_random_graph(n, p=p)

    elif graph_type == "BA":
        d = 2 + int(i >= 150 and n >= 10)
        target_graph = nx.barabasi_albert_graph(n, d)

    else:
        raise ValueError(f"Unknown graph type: {graph_type}")

    metric = "Gram_error_fro"

    return error_by_GED(target_graph, reward_label=metric)


def generate_data_for_regression(args):
    n, data_name, graph_type = args
    output_name = f"{data_name}_for_{n}_nodes.csv"

    tasks = [
        (i, n, graph_type)
        for i in range(300)
    ]

    n_workers = max(1, min(len(tasks), cpu_count() - 4))

    all_data = {}

    with Pool(processes=n_workers) as pool:
        results = pool.imap_unordered(
            generate_single_sample,
            tasks,
        )

        for data in tqdm(
            results,
            total=len(tasks),
            desc=f"Generating data for {data_name} | n={n}",
        ):
            for key, values in data.items():
                all_data.setdefault(key, []).extend(values)

    export_data_e_by_GED(
        all_data,
        output_name,
        graph_type,
    )


if __name__ == "__main__":
    data_name = "data_error_by_GED"
    node_sizes_ER = [x for x in range(7, 13)] + [x for x in range(20, 28, 2)] + [15]
    node_sizes_BA = [x for x in range(7, 13)]
    graph_types = ["ER", "BA"]

    models_to_generate = {
        "ER": node_sizes_ER,
        "BA": node_sizes_BA
    }

    for graph_type, node_sizes in models_to_generate.items():
        for n in node_sizes:
            generate_data_for_regression(
                (n, data_name, graph_type)
            )

        path_data = os.path.join("src", "graph_topology_attacks", "data", 
                                "models_data", graph_type)
        
        all_models = {n_nodes: None for n_nodes in node_sizes}

        for i, n in enumerate(node_sizes):
            file_path = os.path.join(path_data, f"{data_name}_for_{n}_nodes.csv")
            df = pd.read_csv(file_path)
            df = df.dropna().reset_index(drop=True)
            
            count_equilibrage = df['GED'].value_counts().sort_values().iloc[1]
            balanced_idx = (df.groupby('GED', as_index=False, group_keys=False).apply(lambda x: x.sample(min(len(x), count_equilibrage), random_state=42))).index
            df = df.loc[balanced_idx].copy()

            train_df, test_df = split_data(df)

            if data_name == "data_error_by_removing_edges":
                feature_cols = [col for col in df.columns if col not in ["nb_missing_edges", "reward_label"]]
                format_dict = {"y_name": "nb_missing_edges", "xlabel": "Nb. missing edges (Y)"}
            elif data_name == "data_error_by_GED":
                feature_cols = [col for col in df.columns if col not in ["GED", "reward_label", "nb_components"]]
                format_dict = {"y_name": "GED", "xlabel": "GED (Y)"}

            models = {}
            labels = train_df["reward_label"].unique()
            palette = dict(zip(labels, sns.color_palette("tab10", len(labels))))

            for j, label in enumerate(train_df["reward_label"].unique()):
                label_color = palette[label]
                train_l = train_df[train_df["reward_label"] == label]
                
                X_train = train_l[feature_cols]
                y_train = train_l[format_dict["y_name"]]

                model = Pipeline([("scaler", StandardScaler()),
                                ("poly", PolynomialFeatures(degree=2, include_bias=True)),
                                ("Ridge", Ridge(alpha=1.2))
                                ])
                model.fit(X_train, y_train)
                models[label] = model

            all_models[n] = models

        save_dir = os.path.join("prior_models", graph_type)
        os.makedirs(save_dir, exist_ok=True)
        for n, models_n in all_models.items():
            for label, model in models_n.items():
                filename = f"model_n{n}_{label}.pkl"
                joblib.dump(model, os.path.join(save_dir, filename))
                print(f"Model saved: {filename}")
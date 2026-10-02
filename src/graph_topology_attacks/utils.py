#############################################################
#### Import
#############################################################
import json
import networkx as nx
import numpy as np
import time
from matplotlib import pyplot as plt
import joblib
import os

#############################################################
#### Functions
#############################################################
def setting_attacks(G, attackers_mapping: dict, noise=None):
    G = nx.relabel_nodes(G, attackers_mapping)
    N_A = 0
    labels = []
    attackers_neighbors = {}
    for a in attackers_mapping.values():
        non_attacker_neighbors = [n for n in G.neighbors(a) if n not in attackers_mapping.values()]
        list_neighbors = sorted(non_attacker_neighbors, key=lambda x: int(x.split('_')[-1]) if isinstance(x, str) else x)
        attackers_neighbors[a] = list_neighbors
        N_A += len(list_neighbors)
        labels += [f"{a}_to_n{i}" for i in list_neighbors]
    
    private_vectors = {
        node: (
            np.zeros(N_A, dtype=np.float64) if noise is None
            else np.random.normal(loc=0.0, scale=np.sqrt(noise), size=N_A, )
            ) for node in G.nodes()
            }
    nx.set_node_attributes(G, private_vectors, 'x')
    nx.set_node_attributes(G, labels, 'labels')
    return G, attackers_neighbors

def sort_key(item):
    node_id = item[0]

    if isinstance(node_id, str):
        return 0, int(node_id.split('_')[-1])

    return 1, node_id

def first_attack(G, attackers_set):
    a_id = 0
    for a in sorted(attackers_set, key=lambda x: int(x.split('_')[-1])):
        non_attacker_neighbors = [n for n in G.neighbors(a) if n not in attackers_set]
        l_neighbors = sorted(non_attacker_neighbors, key=lambda x: int(x.split('_')[-1]) if isinstance(x, str) else x)
        for id, neighbor in enumerate(l_neighbors):
            G.nodes[neighbor]['x'][a_id+id] = 1.0
        a_id += len(l_neighbors)

def first_DGD_attack(G, attackers_set, W):
    a_id = 0
    for a in sorted(attackers_set, key=lambda x: int(x.split('_')[-1])):
        non_attacker_neighbors = [n for n in G.neighbors(a) if n not in attackers_set]
        l_neighbors = sorted(non_attacker_neighbors, key=lambda x: int(x.split('_')[-1]) if isinstance(x, str) else x)
        for id, neighbor in enumerate(l_neighbors):
            G.nodes[neighbor]['x'][a_id+id] = 1.0
        a_id += len(l_neighbors)

def build_X(G):
    sorted_nodes = sorted(G.nodes(data=True), key=sort_key)
    X = np.vstack([attributes["x"] for _, attributes in sorted_nodes])

    visible_nodes = [i for i, (node_id, _) in enumerate(sorted_nodes) 
                     if str(node_id).startswith('attacker')
                     or any(str(n).startswith('attacker') for n in G.neighbors(node_id))]

    return X, visible_nodes

def compute_gossip_matrix(G, return_mapping=False):
    sorted_degree = sorted(G.degree(), key = lambda x: x[0] if type(x[0])==int else - 1/(1+int(x[0].split('_')[-1])))
    
    mapping = [node for (node, val) in sorted_degree]
    deg = np.array([val for (_, val) in sorted_degree], dtype=float)
    Adj = nx.adjacency_matrix(G, mapping).toarray()
    
    W = deg[:, None] + deg[None, :]
    W[W > 0] = 1 / W[W > 0]
    W = W * Adj

    S = W.sum(axis=1)
    W += np.diag(1 - S)

    if return_mapping:
        return W, mapping
    
    return W

def update_rules(W, H, T):
    H_t = W @ H
    history = [H, H_t]
    for _ in range(T-2): # T=0 & T=1 already computed
        H_t_1 = W @ H_t
        H_t = H_t_1
        history.append(H_t)
    
    return history

def signal_propagation(G, attackers_set, N_A, T, W=None, synchron=False):
    if W is None:
        W = compute_gossip_matrix(G)
        
    if synchron:
        first_DGD_attack(G, attackers_set, W)
    else:
        first_attack(G, attackers_set)

    X, visible_nodes= build_X(G)
    X_history = update_rules(W, X, T)  
    
    return [X_t[visible_nodes, :N_A] for X_t in X_history]

def load_vne_instance(filename):
    with open(filename, 'r') as f:
        instance = json.load(f)
    return instance

def construct_graph_from_vne_instance(instance_path):
    instance = load_vne_instance(instance_path)
    n_of_nodes = instance['n']
    l_edges = [(edges['e'][0], edges['e'][1]) for edges in instance['edges']]

    G = nx.Graph()
    G.add_nodes_from(range(n_of_nodes))
    for u, v in l_edges:
        G.add_edge(u, v)
    return G

def plot_reconstruction(G_infected, G_guess, reward, error, reward_label, number_of_nodes, l=99, N=99, title=None):
    node_mapping = nx.vf2pp_isomorphism(G_infected, G_guess)
    is_iso = False
    if node_mapping is not None:
        is_iso = True
        node_mapping = nx.vf2pp_isomorphism(G_infected, G_guess)
        G_guess = nx.relabel_nodes(G_guess, {v: k for k, v in node_mapping.items()})

    fig = plt.figure(figsize=(19, 7), layout='constrained')
    axs = fig.subplot_mosaic([["target", "guess", "reward"],
                              ["target", "guess", "error"]])

    pos1 = nx.spring_layout(G_infected, seed=42)
    pos2 = nx.spring_layout(G_guess, seed=42)

    def draw_with_colored_labels(G, pos, ax, edge_color="gray", node_size=800, node_color="tab:blue"):
        l_attackers = [n for n in G.nodes() if str(n).startswith("attacker")]
        l_nodes = [n for n in G.nodes() if not str(n).startswith("attacker")]

        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_size,
                               nodelist=l_nodes, 
                               node_color=node_color)
        
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_size, 
                               nodelist=l_attackers, 
                               node_color="tab:red")
        
        if node_color == "tab:blue":
            attacker_nodes = {n for n in G.nodes() if str(n).startswith("attacker")}

            attacker_edges = [(u, v) for u, v in G.edges() if u in attacker_nodes or v in attacker_nodes]
            other_edges    = [(u, v) for u, v in G.edges() if u not in attacker_nodes and v not in attacker_nodes]

            nx.draw_networkx_edges(G, pos, ax=ax, edgelist=other_edges,   alpha=0.5, width=6, edge_color=edge_color)
            nx.draw_networkx_edges(G, pos, ax=ax, edgelist=attacker_edges, width=6, edge_color="black")
        else:
            nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.5, width=6, edge_color=edge_color)

        nx.draw_networkx_labels(G, pos, ax=ax, font_size=10, font_color="white")

    # Graphe 1 : Target
    draw_with_colored_labels(G_infected, pos1, axs["target"], node_color="tab:green")
    axs["target"].set_title(f"Target Graph with {G_infected.number_of_edges()} edges")

    # Graphe 2 : Guess
    draw_with_colored_labels(G_guess, pos2, axs["guess"])
    axs["guess"].set_title(
        f"Reconstructed Graph with {G_guess.number_of_edges()} edges : {reward[-1]:.4f}\n"
        f"Isomorphic test: {is_iso}"
    )

    # Graphe 3.1 : Reward
    axs["reward"].plot(list(range(1, len(reward) + 1)), reward, marker='o', color='orange')
    axs["reward"].set_title("Reward Evolution")
    axs["reward"].set_xlabel("#Edges Added")
    axs["reward"].set_ylabel("Reward: " + r"$e(-L(G_*, \tilde{G})) \ / \ \#\mathrm{comp}(G)$")
    axs["reward"].set_xticks(range(1, len(reward) + 1))

    # Graphe 3.2 : Error
    axs["error"].plot(list(range(1, len(error) + 1)), error, marker='o', color='red')
    axs["error"].set_title(f"Loss {reward_label.replace('_', ' ')} Evolution")
    axs["error"].set_xlabel("#Edges Added")
    axs["error"].set_ylabel(r"$L(G_*, \tilde{G})$")
    axs["error"].set_xticks(range(1, len(error) + 1))

    if title is None:
        title = f"test_{number_of_nodes}_nodes_reward_{reward_label}_{l}_{N}.png"
        
    plt.savefig(title, dpi=300)
    plt.close()


def guess_graph_initialization(n_nodes, init_attackers:dict, G_target=None):
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    guess_a_mapping = {}

    for a_id, a_neighbors in init_attackers.items():
        a_node = int(a_id.split("_")[1])
        guess_a_mapping[a_node] = a_id
        for u in a_neighbors:
            G.add_edge(a_node, u)

    guess_infected, _ = setting_attacks(G, guess_a_mapping)
    if G_target is not None:
        assert set(guess_infected.nodes()).issubset(G_target.nodes()), \
            f"Missing nodes in G_target: \n\
        Target nodes: {list(G_target.nodes())}\n\
        Guess nodes: {list(guess_infected.nodes())}"

        assert all(
            G_target.has_edge(u, v)
            for u, v in guess_infected.edges()
        ), f"Missing edges in G_target: \n\
                Target edges: {list(G_target.edges())}\n\
                Guess edges: {list(guess_infected.edges())}"

    return guess_infected

def load_model(n_nodes, r_label, instance):
    model_name = f"model_n{n_nodes}_{r_label}.pkl"

    try:
        model_path = os.path.join("prior_models", instance, model_name)
        model = joblib.load(model_path)

    except FileNotFoundError:
        l_model = [int(x.replace("model_n", "").split("_")[0]) 
                   for x in os.listdir(os.path.join("prior_models", instance)) 
                   if x != "models_rm_edges"]

        distance = np.abs(np.array(l_model) - n_nodes)
        closest_model = np.argmin(distance)

        model_name = f"model_n{l_model[closest_model]}_{r_label}.pkl"
        model_path = os.path.join("prior_models", instance, model_name)
        model = joblib.load(model_path)

        print(
            f"Model for {n_nodes} nodes and reward '{r_label}' not found. "
            f"Using model with closest #nodes instead: {model_name}"
        )

    return model
#############################################################
#### Import
#############################################################
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

from graph_topology_attacks.utils import signal_propagation, load_model, guess_graph_initialization
from graph_topology_attacks.metrics import matrix_norm, Gram_matrix_norm

import time
from itertools import combinations
import pandas as pd
import random

##############################################################
#### Functions
##############################################################
def reward_function(G_t, X_history, X_guess, metric='Gram_error_fro'):
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
    error = np.array(error_o_time).mean()

    if nx.is_connected(G_t):
        return np.exp(- error), error

    n_components = nx.number_connected_components(G_t)
    return np.exp(- error) / (1 + np.log(n_components)), error


def prior_function(moves_to_compute, model, G_t, X_history, nb_edges:int, 
                   N_a:int, attackers_set, metric='Gram_error_fro', mu=1.0,
                   nb_simu=30, limited_model=False):
    
    prior = {}
    for _ in range(nb_simu):
        all_features = []
        G_prior = G_t.copy()

        while not nx.is_connected(G_prior) and G_prior.number_of_edges() < nb_edges - 1:
            set_of_connected_components = list(nx.connected_components(G_prior))
            attacker_component = [comp for comp in set_of_connected_components if any(str(n).startswith("attacker") for n in comp)][0]
            
            legal_connected_nodes = [n for n in attacker_component if n not in attackers_set]
            legal_disconnected_nodes = [n for n in G_prior.nodes() if n not in attacker_component and n not in attackers_set]
            action = (random.choice(legal_connected_nodes), random.choice(legal_disconnected_nodes))
            G_prior.add_edge(*action)

        possible_edges = [(a, b) for a, b in combinations(G_prior.nodes(), 2) 
                        if not G_prior.has_edge(a, b) and 
                        not (str(a).startswith("attacker") 
                            or 
                            str(b).startswith("attacker"))]
        if possible_edges:
            random.shuffle(possible_edges)
            remaining_edges_to_add = (nb_edges - 1) - G_prior.number_of_edges()
            if remaining_edges_to_add > 0:
                G_prior.add_edges_from(possible_edges[:remaining_edges_to_add])
                latent_edges = possible_edges[remaining_edges_to_add:]
            else:
                latent_edges = possible_edges
        else:
            latent_edges = []

        for m in moves_to_compute:        
            if G_prior.has_edge(*m):
                m = latent_edges[0]
            
            G_prior.add_edge(*m)

            X_guess = signal_propagation(G_prior, attackers_set, N_a, T=len(X_history))
            records = {}
            metric_key = metric
            match metric:
                case 'fro':
                    records = {metric: {t: [] for t in range(len(X_history))}}
                    records = matrix_norm(X_history, X_guess, records, norm_type='fro')
                case 'Gram_error_fro':
                    Gram_matrix_norm(X_history, X_guess, records, norm_type='fro')
                case _:
                    raise ValueError(
                        f"Unsupported reward metric '{metric}'."
                    )

            error = np.array(list(records[metric_key].values()))
            nb_errors_features = len(error) if not limited_model else min(52, len(error))
            features = {
                "nb_edges": nb_edges,
                "nb_neighbors_attackers": N_a,
                **{f"error_{t}": error[t][0] for t in range(nb_errors_features)}
            }
            all_features.append(features)
            
            G_prior.remove_edge(*m)

        if all_features:
            X_infer = pd.DataFrame(all_features)
            ged_preds = model.predict(X_infer)

            for i, m in enumerate(moves_to_compute):
                ged_pred = max(0, ged_preds[i])
                prior_m = prior.get(str(sorted(m)), 0.0)
                prior_m += - (mu / nb_simu) * (ged_pred / nb_edges)
                prior[str(sorted(m))] = prior_m

    return prior


def get_legal_moves(G:nx.Graph):
    nodes_list = [n for n in G.nodes() if not str(n).startswith("attacker")]
    l_moves = [(a, b) for a, b in combinations(nodes_list, 2)
               if not G.has_edge(a, b)]
    
    l_moves = list(set(tuple(sorted(action)) for action in l_moves))
    
    return l_moves
    

def play(G: nx.Graph, curr_node:int, action:int, 
         X_history:np.array, attackers_set:list,
         N_A:int, time_step:int, reward_label:str):

    G.add_edge(curr_node, action)
    X_guess = signal_propagation(G, attackers_set, N_A, T=time_step)
    reward, error = reward_function(G, X_history, X_guess, metric=reward_label)
    done = np.allclose(X_history[:time_step], X_guess[:time_step], atol=1e-3, rtol=0)

    return G, reward, error, done


def playout(policy:dict, prior:dict, G:nx.Graph, X_history:np.array, 
            attackers_set:list, N_A:int, time_step:int, 
            reward_label:str, n_edges:int, tau=1.0):
    
    G_bis = G.copy()
    sequence = []
    done = False

    key = ""
    reward = 0.0

    legal_moves = get_legal_moves(G_bis)
    while True:

        if done or (n_edges == G_bis.number_of_edges()):
            return reward, sequence

        z = 0.0
        logits_vect = []
        actions_vect = []

        for m in legal_moves:
            key_child = key + "," + str(m)
            p_m = policy.get(key_child, 0.0)
            logit = (p_m/tau) + prior[str(sorted(m))]

            actions_vect.append(m)
            logits_vect.append(logit)

        if  len(actions_vect) == 0:
            return 0, []

        logits_vect = np.asarray(logits_vect, dtype=np.float64)
        shifted_logits = logits_vect - np.max(logits_vect)
        weights_vect = np.exp(shifted_logits)
        z = np.sum(weights_vect)

        action_idx = np.random.choice(len(actions_vect), p=weights_vect / z)
        curr_node, action = actions_vect[action_idx]

        G_bis, reward, e, done = play(G_bis, curr_node, action, X_history, attackers_set, N_A, time_step, reward_label)
        legal_moves.remove((curr_node, action))
        sequence.append((curr_node, action))
        key += "," + str((curr_node, action))


def adapt(policy, prior, sequence, G: nx.Graph, X_history: np.array, 
          attackers_set: list, nb_edges: int, N_A: int, time_step: int, 
          reward_label: str, tau=1.0, alpha=1.0):

    G_bis = G.copy()
    polp = policy.copy()
    key = ""
    moves = get_legal_moves(G_bis)

    for action in sequence:

        key_best = key + "," + str(action)
        polp[key_best] = polp.get(key_best, 0.0) + alpha

        logits = []

        for move in moves:
            key_child = key + "," + str(move)
            policy_value = policy.get(key_child, 0.0)
            prior_value = prior[str(sorted(move))]
            logit = policy_value / tau + prior_value

            policy[key_child] = logit
            logits.append(logit)

        logits = np.asarray(logits, dtype=np.float64)

        weights = np.exp(logits - np.max(logits))
        probabilities = weights / np.sum(weights)

        for index, move in enumerate(moves):
            key_child = key + "," + str(move)
            polp[key_child] = polp.get(key_child, 0.0)

            polp[key_child] -= (alpha / tau) * (probabilities[index] - int(sorted(move) == sorted(action)))

        G_bis, *_ = play(G_bis, action[0], action[1], X_history, 
                         attackers_set, N_A, time_step, reward_label )

        moves.remove(action)
        key += "," + str(action)

    return polp


def NRPA(
         G:nx.Graph, 
         X_history:np.array, 
         n_edges:int,
         attackers_set:list, 
         N_A:int, 
         time_step:int,
         level:int, 
         N:int,
         policy:dict,
         prior:dict,
         reward_label:str):

    global start_time, time_budget, global_best_score

    if time.time() - start_time >= time_budget:
        return global_best_score, []
    
    if level == 0:
        score, sequence = playout(policy, prior, G.copy(), X_history, 
                                  attackers_set, N_A, time_step, reward_label, n_edges, tau=1.0)
        
        if score > global_best_score:
            global_best_score = score

        return score, sequence
    else:
        best_score = float("-inf")
        best_seq =[]
        for _ in range(N):

            if time.time() - start_time >= time_budget:
                break

            reward, sequence = NRPA(G.copy(), X_history, n_edges, attackers_set, 
                                    N_A, time_step, level - 1, N, policy, prior, reward_label)

            if reward == 1.0:
                if reward > best_score:
                    best_score = reward
                    
                return best_score, sequence
            
            if reward > best_score:
                best_score = reward
                best_seq = sequence
                policy = adapt(policy, prior, sequence, G, X_history, attackers_set, 
                               n_edges, N_A, time_step, reward_label, tau=1.0, alpha=1.1)

        return best_score, best_seq


def run(number_of_nodes:int,
        number_of_edges:int,
        X_history:np.array,
        attackers_set:list, 
        N_A:int,
        init_attackers:dict, 
        time_step:int,
        level:int,
        N:int,
        reward_label:str,
        budget:float,
        instance_name:str = None,
        mu=1.0):

    global start_time, time_budget, global_best_score

    start_time = time.time()
    time_budget = budget
    global_best_score = float("-inf")

    policy = {}
    prior = {}
    score_history = []
    error_history = []

    model = load_model(n_nodes=number_of_nodes, 
                       r_label=reward_label, 
                       instance=instance_name)
            
    time_start = time.time()
    G = guess_graph_initialization(number_of_nodes,
                                   init_attackers)
    G_bis = G.copy()

    moves_to_compute = [sorted((a, b)) for a, b in combinations(G_bis.nodes(), 2) if
                        not (str(a).startswith("attacker") or str(b).startswith("attacker")) and 
                        not G_bis.has_edge(a, b)]    
    prior = prior_function(moves_to_compute, model, G_bis, X_history, number_of_edges, 
                           N_A, attackers_set, metric=reward_label, mu=mu)
    
    score, sequence = NRPA(G_bis, X_history, number_of_edges, 
                            attackers_set, N_A, time_step, level, N, 
                            policy, prior, reward_label)

    for action in sequence:
        G_bis, reward, e, done = play(G_bis, action[0], action[1], X_history, attackers_set, 
                                      N_A, time_step, reward_label)
        score_history.append(reward)
        error_history.append(e)

    time_end = time.time()

    return G_bis, score_history, error_history, f"{time_end - time_start:.2f} seconds"


def plot_combined_results(results, block_size):
    results_sorted = sorted(results, key=lambda r: r["n_attackers"])

    r0 = results_sorted[0]
    fig, ax = plt.subplots(figsize=(12, 5))

    cmap = plt.get_cmap("tab10")

    for idx, r in enumerate(results_sorted):
        means_arr = np.array(r["means"])
        stds_arr  = np.sqrt(np.array(r["vars_"]))
        x = range(1, len(means_arr) + 1)
        color = cmap(idx / max(len(results_sorted) - 1, 1))

        label = f"{r['n_attackers']}"
        ax.plot(x, means_arr, label=label, color=color)
        ax.fill_between(x,
                        means_arr - stds_arr,
                        means_arr + stds_arr,
                        alpha=0.15, color=color)

    ax.set_ylim(0.995, 1.002)
    ax.set_yscale("log")
    ax.set_xlabel(f"Block (size {block_size})")
    ax.set_ylabel("Mean score (log scale)")
    ax.set_title(f"Mean score per block — L={r0['l']} | N={r0['N']} | {r0['n_nodes']} noeuds")
    legend = ax.legend(title="# Attackers", ncol=2, fontsize=8, loc='upper right', 
              bbox_to_anchor=(1.15, 1), bbox_transform=ax.transAxes)
    text = ""
    for r in sorted(results, key=lambda r: r["n_attackers"]):
        text += (
            f"#A={r['n_attackers']:2d}\n"
            f"R={r['final_reward']:.4f} | "
            f"T= {r['time_taken'].replace(' seconds', 's')}\n\n"
            )
    
    fig.canvas.draw()
    legend_bbox = legend.get_window_extent(fig.canvas.get_renderer())
    legend_bbox_ax = legend_bbox.transformed(ax.transAxes.inverted())

    ax.text(
        legend_bbox_ax.x0 + legend_bbox_ax.width / 2, 
        legend_bbox_ax.y0 - 0.15,
        text,
        transform=ax.transAxes,
        ha='center', va='top', fontsize=8,
        family='monospace' 
    )

    plt.axhline(y=1, color='black', linestyle='--')
    plt.tight_layout()
    fname = f"Score_evolution_all_attackers_n{r0['n_nodes']}_L{r0['l']}_N{r0['N']}.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"Plot combiné sauvegardé : {fname}")
#############################################################
#### Import
#############################################################
import networkx as nx
import numpy as np
from graph_topology_attacks.utils import signal_propagation, guess_graph_initialization
from graph_topology_attacks.metrics import matrix_norm, Gram_matrix_norm
import time
from itertools import combinations

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


def get_legal_moves(G:nx.Graph):
    nodes_list = [n for n in G.nodes() if not str(n).startswith("attacker")]
    
    return [(a, b)
            for a, b in combinations(nodes_list, 2)
            if not G.has_edge(a, b)
            ]
    

def play(G: nx.Graph, curr_node:int, action:int, 
         X_history:np.array, attackers_set:list, 
         N_A:int, time_step:int, reward_label:str):

    G.add_edge(curr_node, action)
    X_guess = signal_propagation(G, attackers_set, N_A, T=time_step)
    reward, error = reward_function(G, X_history, X_guess, metric=reward_label)
    done = np.allclose(X_history[:time_step], X_guess[:time_step], atol=1e-3, rtol=0)

    return G, reward, error, done


def playout(policy:dict, G:nx.Graph, X_history:np.array, 
            attackers_set:list, N_A:int, time_step:int, 
            reward_label:str, n_edges:int, epsilon=1e-6):
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
        weights_vect = []
        actions_vect = []

        for m in legal_moves:
            key_child = key + "," + str(m)
            p_m = policy.get(key_child, 0.0)
            ex = np.exp(p_m)
            z += ex
            actions_vect.append(m)
            weights_vect.append(ex)


        if  len(actions_vect) == 0:
            return 0, []

        action_idx = np.random.choice(len(actions_vect), p=np.array(weights_vect)/z)
        curr_node, action = actions_vect[action_idx]

        G_bis, reward, e, done = play(G_bis, curr_node, action, X_history, attackers_set, N_A, time_step, reward_label)
        legal_moves.remove((curr_node, action))
        sequence.append((curr_node, action))
        key += "," + str((curr_node, action))


def adapt(policy, sequence, G:nx.Graph, X_history:np.array, attackers_set:list, 
          N_A:int, time_step:int, reward_label:str, epsilon=1e-6, alpha=1.0):
    G_bis = G.copy()
    polp = policy.copy()
    key = ""
    moves = get_legal_moves(G_bis)
    for action in sequence:

        key_best = key + "," + str(action)
        polp[key_best] = polp.get(key_best, 0.0)
        polp[key_best] += alpha

        z = 0.0
        for m in moves:
            key_child = key + "," + str(m)
            policy[key_child] = policy.get(key_child, 0.0)
            z += np.exp(policy[key_child])

        for m in moves:
            key_child = key + "," + str(m)
            polp[key_child] = polp.get(key_child, 0.0)
            p_m = policy[key_child]
            polp[key_child] -= alpha * (np.exp(p_m) / z)

        G_bis, *_ = play(G_bis, action[0], action[1], X_history, attackers_set, N_A, time_step, reward_label)
        moves.remove(action)
        key += "," + str(action)

    return polp


def NRPA(G:nx.Graph, 
         X_history:np.array, 
         n_edges:int,
         attackers_set:list, 
         N_A:int, 
         time_step:int,
         level:int, 
         N:int,
         policy:dict,
         reward_label:str):

    global start_time, time_budget, global_best_score

    if time.time() - start_time >= time_budget:
        return global_best_score, []
    
    if level == 0:
        score, sequence = playout(policy, G.copy(), X_history, attackers_set, N_A, time_step, reward_label, n_edges)
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
                                    N_A, time_step, level - 1, N, policy, reward_label)

            if reward == 1.0:
                if reward > best_score:
                    best_score = reward
                    
                return best_score, sequence
            
            if reward > best_score:
                best_score = reward
                best_seq = sequence
                
                policy = adapt(policy, sequence, G, X_history, attackers_set, N_A, 
                               time_step, reward_label, alpha=1.0)

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
        budget:float):
    
    global start_time, time_budget, global_best_score

    start_time = time.time()
    time_budget = budget
    global_best_score = float("-inf")

    policy = {}
    score_history = []
    error_history = []
    time_start = time.time()

    G = guess_graph_initialization(number_of_nodes,
                                   init_attackers)
    
    G_bis = G.copy()
    score, sequence = NRPA(G_bis, X_history, number_of_edges, 
                            attackers_set, N_A, time_step, level, N, 
                            policy, reward_label)
        
    for action in sequence:
        G_bis, reward, e, done = play(G_bis, action[0], action[1], X_history, attackers_set, 
                                    N_A, time_step, reward_label)
        score_history.append(reward)
        error_history.append(e)
        
    time_end = time.time()
    return G_bis, score_history, error_history, f"{time_end - time_start:.2f} seconds"
#############################################################
#### Import
#############################################################
import networkx as nx
import numpy as np
import time
from numpy import linalg as LA
from scipy.linalg import eigvalsh
from graph_topology_attacks.utils import  compute_gossip_matrix
from sklearn.metrics.pairwise import cosine_similarity, polynomial_kernel, sigmoid_kernel

#############################################################
#### Functions
#############################################################
def matrix_norm(X_target, X_guess, records=None, norm_type='fro'):
    if records is None:
        records = {}
    time_step = len(X_guess)
    n_rows = X_target[0].shape[0]
    for t in range(time_step):
        score = float(LA.norm(X_target[t] - X_guess[t], norm_type)/(n_rows))
        records[norm_type][t].append(round(score, 8))
                
    return records

def graph_edit_paths(G1, G2, records=None, t_out=300):
    if records is None:
        records = {}

    t_start = time.time()
    dist = nx.graph_edit_distance(G1, G2, timeout=t_out)
    t_end = time.time()
    
    ged = records.get("ged", [])
    
    if t_end - t_start >= t_out:
        ged.append((dist, 1))
    else:
        ged.append((dist, 0))

    records["ged"] = ged

def second_gossip_eigenvalue(graph, target_lambda_2, records=None):
    if records is None:
        records = {}
    W = compute_gossip_matrix(graph)
    vals = np.linalg.eigvalsh(W)
    lbd_2 = sorted(vals, reverse=True)[1]
    lambda_2_diff = records.get("lambda_2_diff", [])
    r = target_lambda_2 - lbd_2
    lambda_2_diff.append(r)
    records["lambda_2_diff"] = lambda_2_diff

    
def spectral_gap(lambda_2):
    return 1 - lambda_2


def Gram_matrix_norm(X_target, X_guess, records=None, norm_type='fro'):
    if records is None:
        records = {}
    time_step = len(X_guess)
    gef = records.get("Gram_error_" + norm_type, {t: [] for t in range(time_step)})
    for t in range(time_step):
        target_gram = X_target[t].T @ X_target[t]
        guess_gram = X_guess[t].T @ X_guess[t]
        n_rows = target_gram.shape[0]
        score = float(LA.norm(target_gram - guess_gram, norm_type)/(n_rows))
        gef[t].append(round(score, 8))

    records["Gram_error_" + norm_type] = gef


def Gram_matrix_D_rieman(X_target, X_guess, records=None, epsilon=1e-6):
    if records is None:
        records = {}
    time_step = len(X_guess)
    gdr = records.get("Gram_error_riemann", {t: [] for t in range(time_step)})
    for t in range(time_step):
        target_gram = X_target[t].T @ X_target[t]
        guess_gram = X_guess[t].T @ X_guess[t]
        n_rows = target_gram.shape[0]
        
        # Régularisation pour garantir SPD
        target_gram_reg = target_gram + epsilon * np.eye(n_rows)
        guess_gram_reg  = guess_gram  + epsilon * np.eye(n_rows)
        
        # Valeurs propres généralisées
        eigenvalues = eigvalsh(target_gram_reg, guess_gram_reg)
        
        # Distance Riemannienne affine-invariante normalisée
        score = float(np.sqrt((np.log(eigenvalues) ** 2).sum()) / n_rows)
        gdr[t].append(round(score, 8))
    
    records["Gram_error_riemann"] = gdr

def l1_degree_distance(G1, G2, records=None):
    if records is None:
        records = {}
    degree_dist_G1 = np.array(sorted([deg for n, deg in G1.degree()]))
    degree_dist_G2 = np.array(sorted([deg for n, deg in G2.degree()]))
    
    dist = np.mean(np.abs(degree_dist_G1 - degree_dist_G2))
    
    l1_distances = records.get("l1_degree_distance", [])
    l1_distances.append(dist)
    records["l1_degree_distance"] = l1_distances

def kernel_based_reward(X_target, X_guess, records=None, norm_type='cosine'):
    if records is None:
        records = {}
    function_kernel = {
        'cosine': cosine_similarity,
        'polynomial': polynomial_kernel,
        'sigmoid': sigmoid_kernel
    }
    
    f_kernel = function_kernel[norm_type]
    time_step = len(X_guess)
    gef = records.get(norm_type, {t: [] for t in range(time_step)})
    for t in range(time_step):
        # n_rows = X_target[t].shape[0]
        score = float(f_kernel(X_target[t], X_guess[t]).mean())
        gef[t].append(round(score, 8))
    
    records[norm_type] = gef

def Laplacian_Spectral_Distance(G_target, G_guess, records=None):
    if records is None:
        records = {}
    L_target = nx.laplacian_matrix(G_target).toarray()
    L_guess = nx.laplacian_matrix(G_guess).toarray()
    eig_target = np.sort(LA.eigvalsh(L_target))
    eig_guess = np.sort(LA.eigvalsh(L_guess))
    distance = LA.norm(eig_target - eig_guess)

    laplacian_distances = records.get("laplacian_spectral_distance", [])
    laplacian_distances.append(distance)
    records["laplacian_spectral_distance"] = laplacian_distances

def Spectral_Radius_Distance(G_target, G_guess, records=None):
    if records is None:
        records = {}
    A_target = nx.adjacency_matrix(G_target).toarray()
    A_guess = nx.adjacency_matrix(G_guess).toarray()
    eig_target = np.max(np.abs(LA.eigvals(A_target)))
    eig_guess = np.max(np.abs(LA.eigvals(A_guess)))
    distance = abs(eig_target - eig_guess)

    spectral_radius_distances = records.get("spectral_radius_distance", [])
    spectral_radius_distances.append(distance)
    records["spectral_radius_distance"] = spectral_radius_distances

def convergence_speed(X):
    X = np.array(X)
    return np.sum(LA.norm(X[:-1] - X[1:], 'fro', axis=(1, 2)))

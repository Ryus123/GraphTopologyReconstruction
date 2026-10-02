#############################################################
#### Import
#############################################################
import argparse
import csv
import json
import os
import pickle
import random
import time
from itertools import combinations
from multiprocessing import Pool, cpu_count
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import networkx as nx
import numpy as np

from graph_topology_attacks.utils import signal_propagation, setting_attacks, guess_graph_initialization, load_model
from graph_topology_attacks.algo.offline_prior import prior_function, get_legal_moves, play
from graph_topology_attacks.graph_generator import graph_to_json

##############################################################
#### Variables
##############################################################
start_time = 0.0
time_budget = 0.0
global_best_score = float("-inf")
score_history = []

##############################################################
#### Functions
##############################################################
def initialized_prior(moves):
    return {str(sorted(move)): 0.0 for move in moves}


def current_prior(prior_state):
    return prior_state["current"]


def maybe_update_unknown_prior( prior_state, best_sequence, base_graph, 
                               X_history, N_A, attackers_set, reward_label,):
    """Update the unknown-m prior only when the estimated edge count changes."""
    if prior_state["mode"] != "unknown" or not best_sequence:
        return False

    estimated_edges = base_graph.number_of_edges() + len(best_sequence)

    if estimated_edges == prior_state["current_edge_estimate"]:
        return False

    if estimated_edges in prior_state["cache"]:
        prior_state["current"] = prior_state["cache"][estimated_edges]
        prior_state["current_edge_estimate"] = estimated_edges
        prior_state["update_log"].append(
            {
                "time": time.time() - start_time,
                "estimated_edges": estimated_edges,
                "source": "cache",
            }
        )
        return True

    maximum_edges = base_graph.number_of_edges() + len(
        get_legal_moves(base_graph)
    )
    if not base_graph.number_of_nodes() - 1 <= estimated_edges <= maximum_edges:
        return False

    updated_prior = prior_function(
        prior_state["moves_to_compute"],
        prior_state["model"],
        base_graph,
        X_history,
        estimated_edges,
        N_A,
        attackers_set,
        metric=reward_label,
        nb_simu=prior_state["prior_simulations"],
        limited_model=True,
    )

    prior_state["cache"][estimated_edges] = updated_prior
    prior_state["current"] = updated_prior
    prior_state["current_edge_estimate"] = estimated_edges
    prior_state["update_log"].append({
        "time": time.time() - start_time,
        "estimated_edges": estimated_edges,
        "source": "computed",
    })
    return True


def playout( policy, prior_state, G, X_history, attackers_set, N_A, 
            time_step, reward_label, n_edges, edge_count_known, 
            tau=1.0, patience=5, ):
    G_bis = G.copy()
    sequence = []
    best_sequence = []
    key = ""
    reward = 0.0
    best_reward = float("-inf")
    previous_reward = float("-inf")
    stagnation_counter = 0
    legal_moves = get_legal_moves(G_bis)

    while legal_moves:
        if edge_count_known and G_bis.number_of_edges() >= n_edges:
            return reward, sequence

        prior = current_prior(prior_state)
        actions = []
        logits = []

        for move in legal_moves:
            key_child = key + "," + str(move)
            actions.append(move)
            logits.append(
                policy.get(key_child, 0.0) / tau
                + prior.get(str(sorted(move)), 0.0)
            )

        logits = np.asarray(logits, dtype=float)
        logits -= np.max(logits)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum()
        action = actions[np.random.choice(len(actions), p=probabilities)]

        G_bis, reward, _, done = play( G_bis, action[0], action[1], X_history, 
                                      attackers_set, N_A, time_step, reward_label, )

        legal_moves.remove(action)
        sequence.append(action)
        key += "," + str(action)

        if reward > best_reward:
            best_reward = reward
            best_sequence = sequence.copy()

        if edge_count_known:
            if done or G_bis.number_of_edges() >= n_edges:
                return reward, sequence
        else:
            if reward > previous_reward:
                previous_reward = reward
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            if (
                stagnation_counter >= patience
                and G_bis.number_of_edges() >= G_bis.number_of_nodes()
            ):
                break

    if best_reward == float("-inf"):
        return reward, sequence
    return best_reward, best_sequence


def adapt( policy, prior_state, sequence, G, X_history, 
          attackers_set, nb_edges, N_A, time_step, 
          reward_label, edge_count_known, tau=1.0, alpha=1.0, ):
    
    G_bis = G.copy()
    adapted_policy = policy.copy()
    key = ""
    moves = get_legal_moves(G_bis)

    for action in sequence:
        available_moves = list(moves)
        prior = current_prior(prior_state)
        logits = np.asarray(
            [
                policy.get(key + "," + str(move), 0.0) / tau
                + prior.get(str(sorted(move)), 0.0)
                for move in available_moves
            ],
            dtype=float,
        )
        logits -= np.max(logits)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum()

        for move, probability in zip(available_moves, probabilities):
            key_child = key + "," + str(move)
            indicator = int(sorted(move) == sorted(action))
            adapted_policy[key_child] = (
                adapted_policy.get(key_child, 0.0)
                + (alpha / tau) * (indicator - probability)
            )

        G_bis, *_ = play(
            G_bis,
            action[0],
            action[1],
            X_history,
            attackers_set,
            nb_edges if edge_count_known else 0,
            N_A,
            time_step,
            reward_label,
        )
        moves.remove(action)
        key += "," + str(action)

    return adapted_policy


def NRPA(G, X_history, n_edges, attackers_set, N_A, 
         time_step, level, N, policy, prior_state, 
         reward_label, edge_count_known, patience, ):
    global start_time, time_budget, global_best_score, score_history

    if time.time() - start_time >= time_budget:
        return global_best_score, [], score_history

    if level == 0:
        score, sequence = playout( policy, prior_state, G.copy(), 
                                  X_history, attackers_set, N_A, time_step, 
                                  reward_label, n_edges=n_edges, 
                                  edge_count_known=edge_count_known, 
                                  tau=1.0, patience=patience, )

        if score > global_best_score:
            global_best_score = score
            score_history.append((time.time() - start_time, global_best_score))

        return score, sequence, score_history

    local_best_score = float("-inf")
    best_sequence = []
    current_policy = policy.copy()

    for _ in range(N):
        if time.time() - start_time >= time_budget:
            break

        reward, sequence, _ = NRPA( G.copy(), X_history, n_edges, attackers_set, 
                                   N_A, time_step, level - 1, N, current_policy, 
                                   prior_state, reward_label, edge_count_known, 
                                   patience, )

        if reward > local_best_score:
            local_best_score = reward
            best_sequence = sequence

        if not edge_count_known and best_sequence:
            maybe_update_unknown_prior(prior_state, best_sequence, G, X_history, 
                                       N_A, attackers_set, reward_label, )

        if best_sequence:
            current_policy = adapt(current_policy, prior_state, best_sequence, 
                                   G, X_history, attackers_set, n_edges, N_A, 
                                   time_step, reward_label, edge_count_known, 
                                   tau=1.0, alpha=0.8, )

        if reward >= 1.0:
            return reward, sequence, score_history

    return local_best_score, best_sequence, score_history


def atomic_pickle_dump(value, destination):
    destination = Path(destination)
    temporary = destination.with_suffix(
        destination.suffix + f".{os.getpid()}.tmp"
    )
    with temporary.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, destination)


def compute_and_store_partial(task):
    global start_time, time_budget, global_best_score, score_history

    (
        edge_count_mode,
        run_index,
        G_initial,
        X_history,
        true_number_of_edges,
        attackers_set,
        N_A,
        time_step,
        level,
        N,
        initial_prior,
        adaptive_context,
        reward_label,
        budget,
        patience,
        seed,
        partial_path,
        target_graph,
    ) = task

    partial_path = Path(partial_path)
    if partial_path.exists():
        return str(partial_path)

    random.seed(seed)
    np.random.seed(seed)
    start_time = time.time()
    time_budget = budget
    global_best_score = float("-inf")
    score_history = []

    edge_count_known = edge_count_mode == "known"
    prior_state = {
        "mode": edge_count_mode,
        "current": initial_prior.copy(),
        "current_edge_estimate": (true_number_of_edges if edge_count_known else None ),
        "cache": ({true_number_of_edges: initial_prior.copy()} 
                  if edge_count_known else {}),
        "update_log": [],
        **adaptive_context,
    }

    _, best_sequence, history = NRPA(G_initial.copy(), X_history, true_number_of_edges, 
                                     attackers_set, N_A, time_step, level, N, {}, 
                                     prior_state, reward_label, edge_count_known, patience, )

    reconstructed = G_initial.copy()
    reconstructed.add_edges_from(best_sequence)

    partial = {
        "edge_count_mode": edge_count_mode,
        "run_index": run_index,
        "seed": seed,
        "history": list(history),
        "best_sequence": best_sequence,
        "estimated_edge_count": reconstructed.number_of_edges(),
        "true_edge_count": true_number_of_edges,
        "prior_update_log": prior_state["update_log"],
        "prior_cache_edge_counts": sorted(prior_state["cache"].keys()),
        "isomorphic": bool(nx.is_isomorphic(reconstructed, target_graph)),
        "runtime_seconds": time.time() - start_time,
    }
    atomic_pickle_dump(partial, partial_path)
    return str(partial_path)


def prepare_configuration(target_graph, args, partials_directory):
    attacker_node = 0
    attacker_mapping = {attacker_node: "attacker_0"}
    infected_graph, attacker_neighbors = setting_attacks(target_graph, attacker_mapping)
    init_attackers = {"attacker_0": list(nx.neighbors(infected_graph, "attacker_0"))}
    N_A = sum(len(neighbors) for neighbors in attacker_neighbors.values())
    attackers_set = list(attacker_neighbors.keys())
    time_step = 2 * args.nodes
    X_history = signal_propagation(infected_graph, attackers_set, N_A, time_step)
    initial_graph = guess_graph_initialization(args.nodes, init_attackers)

    moves_to_compute = [sorted((u, v)) for u, v in combinations(initial_graph.nodes(), 2)
                        if not (str(u).startswith("attacker") or str(v).startswith("attacker") ) 
                        and not initial_graph.has_edge(u, v)]

    model = load_model(args.nodes, args.reward_label, "ER")

    known_prior = prior_function(moves_to_compute, model, initial_graph, 
                                 X_history, target_graph.number_of_edges(), 
                                 N_A, attackers_set, metric=args.reward_label, 
                                 nb_simu=args.prior_simulations, limited_model=True, )
    unknown_prior = initialized_prior(moves_to_compute)

    adaptive_context = {
        "model": model,
        "moves_to_compute": moves_to_compute,
        "prior_simulations": args.prior_simulations,
    }

    tasks = []
    for mode in ("known", "unknown"):
        initial_prior = known_prior if mode == "known" else unknown_prior
        for run_index in range(args.runs):
            partial_path = partials_directory / f"edge_count_{mode}_run_{run_index:04d}.pkl"
            tasks.append(
                (mode, run_index, initial_graph, X_history, 
                target_graph.number_of_edges(), attackers_set, 
                N_A, time_step, args.level, args.iterations, 
                initial_prior, adaptive_context, args.reward_label, 
                args.time_budget, args.patience, args.seed + run_index, 
                str(partial_path), infected_graph, )
                )

    configuration = {
        "N_A": N_A,
        "initial_graph": initial_graph,
        "infected_target_graph": infected_graph,
    }
    return tasks, configuration


def build_log_time_score_matrix(results, time_grid, initial_score=0.0):
    matrix = np.full((len(results), len(time_grid)), initial_score, dtype=float)
    best_sequences = []

    for run_index, result in enumerate(results):
        history = sorted(result["history"], key=lambda item: item[0])
        best_sequences.append(result["best_sequence"])
        current_score = initial_score
        history_index = 0

        for time_index, current_time in enumerate(time_grid):
            while (
                history_index < len(history)
                and history[history_index][0] <= current_time
            ):
                current_score = max(current_score, history[history_index][1])
                history_index += 1
            matrix[run_index, time_index] = current_score

    return matrix, best_sequences


def first_target_time(history, objective):
    for elapsed, score in sorted(history, key=lambda item: item[0]):
        if score >= objective:
            return float(elapsed)
    return np.nan


def isomorphism_rate(sequences, target_graph, initial_graph):
    outcomes = []
    for sequence in sequences:
        reconstructed = initial_graph.copy()
        reconstructed.add_edges_from(sequence)
        outcomes.append(nx.is_isomorphic(reconstructed, target_graph))
    return float(np.mean(outcomes)) if outcomes else 0.0


def plot_edge_count_comparison(all_results, time_grid, objective, output_path):
    figure, axis = plt.subplots(figsize=(3.5, 2.8))
    styles = {
        "known": {"color": "#1473E6", "label": "Known edge count"},
        "unknown": {"color": "#E05A47", "label": "Unknown edge count"},
    }
    minimum_score = objective

    for mode in ("known", "unknown"):
        matrix = all_results[mode]["score_matrix"]
        iso_rate = all_results[mode]["isomorphism_rate"]
        median = np.median(matrix, axis=0)
        q1 = np.quantile(matrix, 0.25, axis=0)
        q3 = np.quantile(matrix, 0.75, axis=0)
        minimum_score = min(minimum_score, float(np.min(q1)))

        axis.fill_between(time_grid, q1, q3, color=styles[mode]["color"], alpha=0.14, linewidth=0,)
        axis.plot(time_grid, median, color=styles[mode]["color"], linewidth=2.0, 
                  label=(f"{styles[mode]['label']}, "
                         f"iso. {100 * iso_rate:.0f}%"),
        )

    axis.axhline(objective, color="0.25", linestyle="--", linewidth=1.0, label="Target reward", )
    margin = max(0.002, 0.08 * (objective - minimum_score))
    axis.set_ylim(minimum_score - margin, objective + margin)
    axis.set_xscale("log")
    axis.set_xlabel("Wall-clock time (s)", fontfamily="serif", fontsize=10, )
    axis.set_ylabel("Best reconstruction reward", fontfamily="serif", fontsize=10,)
    axis.grid(True, which="major", color="0.88", linewidth=0.7)
    axis.grid(True, which="minor", axis="x", color="0.93", linewidth=0.5)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, 
                frameon=False, fontsize=9, prop={ "family": "serif", "size": 9, },)
    figure.subplots_adjust(left=0.13, right=0.98, top=0.97, bottom=0.25)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    figure.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)

def covers_entire_graph(target_graph, attacker_nodes):
    covered_nodes = set(attacker_nodes)
    for attacker in attacker_nodes:
        covered_nodes.update(target_graph.neighbors(attacker))
    return covered_nodes == set(target_graph.nodes())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, default=10)
    parser.add_argument("--edge-probability", type=float, default=1 / 3)
    parser.add_argument("--graph-seed", type=int, default=4332)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--time-budget", type=float, default=3600)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--workers", type=int, default=max(1, cpu_count() - 4))
    parser.add_argument("--time-points", type=int, default=800)
    parser.add_argument("--level", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--prior-simulations", type=int, default=100)
    parser.add_argument("--reward-label", default="Gram_error_fro")
    parser.add_argument("--objective", type=float, default=1.0)
    parser.add_argument("--output", default=os.path.join("paper_results", "unknown_M_edges"))
    return parser.parse_args()


def replot_existing_results(root_directory, objective=1.0, ):
    root_directory = Path(root_directory)

    all_results = {}
    reference_time_grid = None

    for mode in ("known", "unknown"):
        mode_directory = root_directory / f"edge_count_{mode}"
        curves_path = mode_directory / "learning_curves.npz"

        if not curves_path.is_file():
            raise FileNotFoundError(f"Missing learning curves: {curves_path}")

        with np.load(curves_path) as data:
            time_grid = data["time_grid"]
            score_matrix = data["score_matrix"]

        if reference_time_grid is None:
            reference_time_grid = time_grid
        elif not np.allclose(reference_time_grid, time_grid):
            raise ValueError("Known and unknown modes use different time grids.")

        results_path = mode_directory / "results.pkl"

        if results_path.is_file():
            with results_path.open("rb") as handle:
                results = pickle.load(handle)
            iso_rate = float(np.mean([bool(result["isomorphic"]) for result in results ]))
        else:
            iso_rate = float("nan")

        all_results[mode] = {
            "score_matrix": score_matrix,
            "isomorphism_rate": iso_rate,
        }

    output_directory = root_directory / "edge_count_comparison"
    output_directory.mkdir(parents=True, exist_ok=True,)

    plot_edge_count_comparison(all_results=all_results, time_grid=reference_time_grid, 
                               objective=objective,
                               output_path=output_directory / "known_vs_unknown_edge_count.png" )


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    output_directory = Path(os.path.join(args.output, f"n{args.nodes}"))
    partials_directory = output_directory / "edge_count_comparison_partials"
    figures_directory = Path("paper_figures") / os.path.join( "edge_count_comparison", f"n{args.nodes}")
    output_directory.mkdir(parents=True, exist_ok=True)
    partials_directory.mkdir(parents=True, exist_ok=True)
    figures_directory.mkdir(parents=True, exist_ok=True)

    if args.nodes == 15:
        original_graph = nx.florentine_families_graph()
    elif args.nodes == 34:
        original_graph = nx.karate_club_graph()
    else:
        original_graph = nx.gnp_random_graph(args.nodes, args.edge_probability, seed=args.graph_seed, )

    original_labels = list(original_graph.nodes())
    target_graph = nx.convert_node_labels_to_integers(original_graph, first_label=0, ordering="default")
    node_mapping = { new_label: original_label 
                    for new_label, original_label in enumerate(original_labels)}

    attempt = 0
    while not nx.is_connected(target_graph):
        attempt += 1
        if attempt > 30:
            raise ValueError("Unable to generate a connected target graph.")
        target_graph = nx.gnp_random_graph(args.nodes, args.edge_probability, seed=args.graph_seed + attempt, )
        node_mapping = {node: node for node in target_graph.nodes()}

    if covers_entire_graph(target_graph, [0]):
        raise RuntimeError("Node 0 and its direct neighborhood cover the entire graph.")

    with (output_directory / "target_graph.json").open("w", encoding="utf-8") as handle:
        json.dump(graph_to_json(target_graph, args.nodes), handle, indent=4)

    with (output_directory / "node_mapping.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {str(new): str(old) for new, old in node_mapping.items()},
            handle,
            indent=4,
        )

    total_start = time.time()
    all_tasks, configuration = prepare_configuration(target_graph, args, partials_directory)
    pending_tasks = [task for task in all_tasks 
                     if not Path(task[-2]).exists()]
    workers = max(1, min(args.workers, len(pending_tasks) or 1))

    print(
        f"Total computations={len(all_tasks)}, pending={len(pending_tasks)}, "
        f"workers={workers}",
        flush=True,
    )

    if pending_tasks:
        with Pool(processes=workers) as pool:
            for completed_path in pool.imap_unordered(compute_and_store_partial, pending_tasks, chunksize=1):
                print(f"Stored {completed_path}", flush=True)

    time_grid = np.logspace(np.log10(0.1), np.log10(args.time_budget), args.time_points)
    
    all_results = {}
    summary_rows = []
    initial_graph = configuration["initial_graph"]
    infected_target = configuration["infected_target_graph"]

    for mode in ("known", "unknown"):
        partial_paths = sorted(partials_directory.glob(f"edge_count_{mode}_run_*.pkl"))
        if len(partial_paths) != args.runs:
            raise RuntimeError(
                f"{mode}: {len(partial_paths)} partials found, "
                f"expected {args.runs}."
            )

        results = []
        for path in partial_paths:
            with path.open("rb") as handle:
                results.append(pickle.load(handle))
        results.sort(key=lambda item: item["run_index"])

        score_matrix, best_sequences = build_log_time_score_matrix(results, time_grid, initial_score=0.0)
        objective_times = np.asarray([first_target_time(result["history"], args.objective)
                                      for result in results])
        success_mask = np.isfinite(objective_times)
        success_rate = float(np.mean(success_mask))
        median_time = (
            float(np.median(objective_times[success_mask]))
            if np.any(success_mask)
            else np.nan
        )
        iso_rate = isomorphism_rate(best_sequences, infected_target, initial_graph )
        estimated_edge_counts = np.asarray([result["estimated_edge_count"] for result in results ])

        all_results[mode] = {
            "score_matrix": score_matrix,
            "success_rate": success_rate,
            "median_objective_time": median_time,
            "isomorphism_rate": iso_rate,
        }

        mode_directory = output_directory / f"edge_count_{mode}"
        mode_directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            mode_directory / "learning_curves.npz",
            time_grid=time_grid,
            score_matrix=score_matrix,
            objective_times=objective_times,
            estimated_edge_counts=estimated_edge_counts,
        )
        with (mode_directory / "results.pkl").open("wb") as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

        summary_rows.append({
            "edge_count_mode": mode,
            "true_edge_count": target_graph.number_of_edges(),
            "mean_estimated_edge_count": float(np.mean(estimated_edge_counts)),
            "median_estimated_edge_count": float(np.median(estimated_edge_counts)),
            "success_rate": success_rate,
            "median_time_to_objective_seconds": median_time,
            "isomorphism_rate": iso_rate,
        })

    plot_edge_count_comparison(
        all_results,
        time_grid,
        args.objective,
        figures_directory / "known_vs_unknown_edge_count.png",
    )

    with (output_directory / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)

    metadata = {
        "nodes": args.nodes,
        "target_edges": target_graph.number_of_edges(),
        "attacker_nodes": [0],
        "unknown_mode": {
            "initial_prior": "neutral",
            "prior_update": (
                "Recomputed when the incumbent best sequence implies a new "
                "edge-count estimate; cached by estimated edge count."
            ),
            "patience": args.patience,
        },
        "known_mode": {
            "prior_edge_count": target_graph.number_of_edges(),
            "rollout_stopping_rule": "exact target edge count",
        },
        "time_budget_seconds": args.time_budget,
        "runs_per_mode": args.runs,
        "workers": workers,
        "reward_label": args.reward_label,
        "total_runtime_seconds": time.time() - total_start,
    }
    with (output_directory / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=4)

    print(f"Comparison complete. Results: {output_directory.resolve()}", flush=True, )

##############################################################
#### Main
##############################################################
if __name__ == "__main__":
    main()

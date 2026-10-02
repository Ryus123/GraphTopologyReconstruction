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
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations
from multiprocessing import cpu_count
from pathlib import Path

from matplotlib import pyplot as plt
import networkx as nx
import numpy as np

from graph_topology_attacks.utils import setting_attacks, guess_graph_initialization, load_model
from graph_topology_attacks.algo.offline_prior_DGD import prior_function, adapt, playout
from graph_topology_attacks.graph_generator import graph_to_json

#############################################################
#### Functions
#############################################################
start_time = 0.0
time_budget = 0.0
global_best_score = float("-inf")
score_history = []
improvement_history = []


def load_observations_history(dataset_name="cifar10", results_root="paper_results"):
    dataset_directory = Path(results_root) / dataset_name
    observation_path = dataset_directory / "attacker_observations.npz"
    metadata_path = dataset_directory / "metadata.json"

    if not observation_path.is_file():
        raise FileNotFoundError(f"Observation file not found: {observation_path}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    with np.load(observation_path, allow_pickle=False) as data:
        required_keys = {
            "Z_A",
            "probes",
            "neighbor_node_ids",
            "observation_steps",
            "attack_round",
            "T_obs",
            "model_parameters_at_attack",
            "observed_node_ids",
        }
        missing = required_keys.difference(data.files)
        if missing:
            raise KeyError(f"Missing NPZ keys: {sorted(missing)}")

        Z_A = np.asarray(data["Z_A"], dtype=np.float64)
        probes = np.asarray(data["probes"], dtype=np.float64)
        neighbor_node_ids = np.asarray(data["neighbor_node_ids"], dtype=np.int64)
        observation_steps = np.asarray(data["observation_steps"], dtype=np.int64)
        attack_round = int(data["attack_round"])
        T_obs = int(data["T_obs"])


    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    required_metadata = {"nodes", "attacker"}
    missing_metadata = required_metadata.difference(metadata)
    if missing_metadata:
        raise KeyError(f"Missing metadata fields: {sorted(missing_metadata)}")

    n_nodes = int(metadata["nodes"])
    edge_probability = float(metadata.get("edge_probability", 1.0 / 3.0))
    graph_seed = int(metadata.get("graph_seed", 4332))
    attacker_node = int(metadata["attacker"])

    if Z_A.ndim != 3:
        raise ValueError(
            f"Z_A must have shape (time, rows, probes), got {Z_A.shape}"
        )
    if Z_A.shape[0] != len(observation_steps):
        raise ValueError("Z_A and observation_steps have incompatible lengths.")
    if Z_A.shape[0] != T_obs + 1:
        raise ValueError(
            f"Expected T_obs + 1 = {T_obs + 1} observations, "
            f"got {Z_A.shape[0]}."
        )

    print(f"Loaded observations: {observation_path}")
    print(f"Attacker: {attacker_node}")
    print(f"Observed neighbors: {neighbor_node_ids.tolist()}")
    
    X_history = [ np.around(np.abs(Z_A[t].copy()), decimals=3)
                 for t in range(2*n_nodes)
                 ]

    return {
        "X_history": X_history,
        "metadata": metadata,
        "n_nodes": n_nodes,
        "edge_probability": edge_probability,
        "graph_seed": graph_seed,
        "attacker_node": attacker_node,
        "neighbor_node_ids": neighbor_node_ids,
        "observation_steps": observation_steps,
        "attack_round": attack_round,
        "T_obs": T_obs,
        "probes": probes,
    }


def NRPA(
    G,
    X_history,
    n_edges,
    attackers_set,
    N_A,
    time_step,
    level,
    N,
    policy,
    prior,
    reward_label,
    synchron
):
    global start_time, time_budget, global_best_score
    global score_history, improvement_history

    if time.time() - start_time >= time_budget:
        return global_best_score, [], score_history

    if level == 0:
        score, sequence = playout(
            policy,
            prior,
            G.copy(),
            X_history,
            attackers_set,
            N_A,
            time_step,
            reward_label,
            n_edges,
            tau=1.0,
            synchron=synchron
        )

        if score > global_best_score:
            global_best_score = score
            elapsed = time.time() - start_time
            frozen_sequence = [tuple(edge) for edge in sequence]

            score_history.append((elapsed, global_best_score))
            improvement_history.append({
                "time": float(elapsed),
                "score": float(global_best_score),
                "sequence": frozen_sequence,
            })

        return score, sequence, score_history

    local_best_score = float("-inf")
    best_sequence = []
    current_policy = policy.copy()

    for _ in range(N):
        if time.time() - start_time >= time_budget:
            break

        reward, sequence, _ = NRPA(
            G.copy(),
            X_history,
            n_edges,
            attackers_set,
            N_A,
            time_step,
            level - 1,
            N,
            current_policy,
            prior,
            reward_label,
            synchron
        )

        if reward > local_best_score:
            local_best_score = reward
            best_sequence = sequence
            current_policy = adapt(
                current_policy,
                prior,
                best_sequence,
                G,
                X_history,
                attackers_set,
                n_edges,
                N_A,
                time_step,
                reward_label,
                tau=1.0,
                alpha=0.8,
                synchron=synchron
            )

        if reward >= 1.0:
            return reward, sequence, score_history

    return local_best_score, best_sequence, score_history


def graph_from_sequence(initial_graph, sequence):
    graph = initial_graph.copy()
    graph.add_edges_from(sequence)
    return graph


def safe_graph_edit_distance(graph_a, graph_b, timeout=5.0):
    value = nx.graph_edit_distance(graph_a, graph_b, timeout=timeout)
    return None if value is None else float(value)


def single_NRPA_run(arguments):
    global start_time, time_budget, global_best_score
    global score_history, improvement_history

    (
        run_index,
        seed,
        G_initial,
        X_history,
        number_of_edges,
        attackers_set,
        N_A,
        time_step,
        level,
        N,
        prior,
        reward_label,
        budget,
        target_graph,
        synchron
    ) = arguments

    random.seed(seed)
    np.random.seed(seed)
    start_time = time.time()
    time_budget = budget
    global_best_score = float("-inf")
    score_history = []
    improvement_history = []

    _, best_sequence, history = NRPA(
        G_initial.copy(),
        X_history,
        number_of_edges,
        attackers_set,
        N_A,
        time_step,
        level,
        N,
        {},
        prior,
        reward_label,
        synchron
    )

    reconstructed_graph = graph_from_sequence(G_initial, best_sequence)

    # Last five incumbent improvements in chronological order.
    last_five_improvements = []
    for rank, improvement in enumerate(improvement_history[-5:], start=1):
        sequence = [tuple(edge) for edge in improvement["sequence"]]
        candidate_graph = graph_from_sequence(G_initial, sequence)
        last_five_improvements.append({
            "position_among_last_five": rank,
            "time": float(improvement["time"]),
            "score": float(improvement["score"]),
            "sequence": sequence,
            "isomorphic": bool(nx.is_isomorphic(candidate_graph, target_graph)),
            "ged": safe_graph_edit_distance(candidate_graph, target_graph),
        })

    return {
        "run_index": run_index,
        "seed": seed,
        "history": list(history),
        "best_sequence": [tuple(edge) for edge in best_sequence],
        "last_five_improvements": last_five_improvements,
        "number_of_improvements": len(improvement_history),
        "isomorphic": bool(nx.is_isomorphic(reconstructed_graph, target_graph)),
        "ged": safe_graph_edit_distance(reconstructed_graph, target_graph),
        "runtime_seconds": float(time.time() - start_time),
    }


def prepare_reconstruction(
    target_graph,
    X_history,
    synchron,
    attacker_node,
    observed_neighbor_ids,
    args,
):
    attacker_label = f"attacker_{attacker_node}"
    attacker_mapping = {attacker_node: attacker_label}

    infected_graph, attacker_neighbors = setting_attacks(
        target_graph,
        attacker_mapping,
    )

    actual_neighbors = sorted(
        int(node) for node in attacker_neighbors[attacker_label]
    )
    expected_neighbors = sorted(int(node) for node in observed_neighbor_ids)
    if actual_neighbors != expected_neighbors:
        raise ValueError(
            "The target graph does not match the exported observations: "
            f"metadata neighbors={expected_neighbors}, "
            f"target neighbors={actual_neighbors}."
        )

    init_attackers = {
        attacker_label: list(nx.neighbors(infected_graph, attacker_label))
    }
    attackers_set = [attacker_label]
    N_A = len(expected_neighbors)

    initial_graph = guess_graph_initialization(
        target_graph.number_of_nodes(),
        init_attackers,
        infected_graph,
    )


    moves_to_compute = [
        sorted((u, v))
        for u, v in combinations(initial_graph.nodes(), 2)
        if not str(u).startswith("attacker")
        and not str(v).startswith("attacker")
        and not initial_graph.has_edge(u, v)
    ]

    model = load_model(
        n_nodes=target_graph.number_of_nodes(),
        r_label=args.reward_label,
        instance="ER",
    )

    print(f"Observation history objective:\n{X_history}")

    prior = prior_function(
        moves_to_compute,
        model,
        initial_graph,
        X_history,
        target_graph.number_of_edges(),
        N_A,
        attackers_set,
        metric=args.reward_label,
        nb_simu=args.prior_simulations,
        limited_model=True,
        synchron=synchron
    )
    print("Preparation Done ...")
    return infected_graph, initial_graph, attackers_set, N_A, prior


def run_parallel_reconstructions(
    target_graph,
    initial_graph,
    X_history,
    synchron,
    attackers_set,
    N_A,
    prior,
    args,
):
    parameters = [
        (
            run_index,
            args.seed + run_index,
            initial_graph,
            X_history,
            target_graph.number_of_edges(),
            attackers_set,
            N_A,
            len(X_history),
            args.level,
            args.iterations,
            prior,
            args.reward_label,
            args.time_budget,
            target_graph,
            synchron,
        )
        for run_index in range(args.n_runs)
    ]

    requested_workers = args.max_workers or (cpu_count() - 4)
    max_workers = max(1, min(args.n_runs, requested_workers))

    print(
        f"Running {args.n_runs} independent reconstructions "
        f"with {max_workers} workers.",
        flush=True,
    )

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(single_NRPA_run, parameter): parameter[0]
            for parameter in parameters
        }
        for future in as_completed(futures):
            run_index = futures[future]
            try:
                result = future.result()
                results.append(result)
                print(
                    f"Completed run {run_index + 1}/{args.n_runs}: "
                    f"iso={result['isomorphic']}, "
                    f"improvements={result['number_of_improvements']}, "
                    f"runtime={result['runtime_seconds']:.2f}s",
                    flush=True,
                )
            except Exception as error:
                raise RuntimeError(f"NRPA run {run_index} failed") from error

    results.sort(key=lambda item: item["run_index"])
    return results


def build_time_score_matrix(
    results,
    t_min,
    t_max,
    n_points,
    initial_score=0.0,
):
    if not 0 < t_min < t_max:
        raise ValueError("Expected 0 < t_min < t_max.")

    time_grid = np.geomspace(t_min, t_max, n_points)
    score_matrix = np.full(
        (len(results), n_points),
        initial_score,
        dtype=float,
    )

    for run_index, result in enumerate(results):
        history = sorted(result["history"], key=lambda item: item[0])
        current_score = initial_score
        history_index = 0

        for time_index, current_time in enumerate(time_grid):
            while (
                history_index < len(history)
                and history[history_index][0] <= current_time
            ):
                current_score = max(current_score, history[history_index][1])
                history_index += 1
            score_matrix[run_index, time_index] = current_score

    return time_grid, score_matrix

def plot_reward_evolution(
    results,
    t_min,
    t_max,
    n_points,
    output_path,
    confidence_level=0.95,
):
    if not results:
        raise ValueError("The results list is empty.")

    if confidence_level != 0.95:
        raise ValueError(
            "Only the 95% confidence level is supported."
        )

    if t_min <= 0:
        raise ValueError(
            "t_min must be strictly positive for a logarithmic time axis."
        )

    time_grid, score_matrix = build_time_score_matrix(
        results,
        t_min=t_min,
        t_max=t_max,
        n_points=n_points,
        initial_score=0.0,
    )

    n_seeds = score_matrix.shape[0]
    mean_score = np.mean(score_matrix, axis=0)

    if n_seeds > 1:
        standard_error = (
            np.std(score_matrix, axis=0, ddof=1)
            / np.sqrt(n_seeds)
        )

        lower = mean_score - 1.96 * standard_error
        upper = mean_score + 1.96 * standard_error
    else:
        lower = mean_score.copy()
        upper = mean_score.copy()

    lower = np.clip(lower, 0.0, 1.0)
    upper = np.clip(upper, 0.0, 1.0)

    isomorphism_rate = np.mean(
        [
            bool(result["isomorphic"])
            for result in results
        ]
    )

    with plt.rc_context(
    {
        "font.family": "serif",
        "font.serif": [
            "Times New Roman",
            "Times",
            "DejaVu Serif",
        ],
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "lines.linewidth": 1.5,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
):
        figure, axis = plt.subplots(
            figsize=(3.5, 2.8)
        )

        axis.fill_between(
            time_grid,
            lower,
            upper,
            color="#4C78A8",
            alpha=0.20,
            linewidth=0,
            label="95% confidence interval",
            zorder=1,
        )

        axis.plot(
            time_grid,
            mean_score,
            color="#245A8D",
            linewidth=2.4,
            label="Mean best reward",
            zorder=3,
        )

        axis.set_xscale("log")
        axis.set_xlim(t_min, t_max)
        y_min = max(0.0, float(lower.min()))
        axis.set_ylim(y_min, 1.001)

        axis.set_xlabel("Wall-clock time (s)")
        axis.set_ylabel("Best reconstruction reward")

        axis.grid(
            True,
            which="major",
            color="#D9D9D9",
            linewidth=0.7,
            alpha=0.8,
        )

        axis.grid(
            True,
            which="minor",
            axis="x",
            color="#ECECEC",
            linewidth=0.5,
            alpha=0.7,
        )

        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color("#707070")
        axis.spines["bottom"].set_color("#707070")

        axis.tick_params(
            axis="both",
            which="both",
            direction="out",
            color="#707070",
        )

        axis.text(
            0.98,
            0.06,
            (
                "Exact topology recovery: "
                f"{100.0 * isomorphism_rate:.1f}% "
                f"({n_seeds} seeds)"
            ),
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=7,
            color="#333333",
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "white",
                "edgecolor": "#D0D0D0",
                "linewidth": 0.7,
                "alpha": 0.9,
            },
        )

        axis.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
            ncol=3,
            frameon=False,
            handlelength=2.6,
            columnspacing=1.4,
            handletextpad=0.6,
        )

        figure.subplots_adjust(
            left=0.13,
            right=0.98,
            top=0.97,
            bottom=0.25,
        )

        output_path = Path(output_path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        figure.savefig(
            output_path,
            dpi=400,
            bbox_inches="tight",
            facecolor="white",
        )

        figure.savefig(
            output_path.with_suffix(".pdf"),
            bbox_inches="tight",
            facecolor="white",
        )

        plt.close(figure)

    return time_grid, score_matrix


def json_safe_sequence(sequence):
    return [[str(edge[0]), str(edge[1])] for edge in sequence]


def export_last_five_sequences(results, output_directory):
    sequences_directory = output_directory / "last_five_sequences"
    sequences_directory.mkdir(parents=True, exist_ok=True)

    combined_export = []
    flat_rows = []

    for result in results:
        run_export = {
            "run_index": int(result["run_index"]),
            "seed": int(result["seed"]),
            "number_of_improvements": int(result["number_of_improvements"]),
            "final_isomorphic": bool(result["isomorphic"]),
            "final_ged": result["ged"],
            "last_five_improvements": [],
        }

        for item in result["last_five_improvements"]:
            serializable_item = {
                "position_among_last_five": int(
                    item["position_among_last_five"]
                ),
                "time_seconds": float(item["time"]),
                "score": float(item["score"]),
                "sequence": json_safe_sequence(item["sequence"]),
                "isomorphic": bool(item["isomorphic"]),
                "ged": item["ged"],
            }
            run_export["last_five_improvements"].append(serializable_item)
            flat_rows.append({
                "run_index": int(result["run_index"]),
                "seed": int(result["seed"]),
                "position_among_last_five": serializable_item[
                    "position_among_last_five"
                ],
                "time_seconds": serializable_item["time_seconds"],
                "score": serializable_item["score"],
                "isomorphic": serializable_item["isomorphic"],
                "ged": serializable_item["ged"],
                "sequence": json.dumps(serializable_item["sequence"]),
            })

        run_path = sequences_directory / (
            f"run_{result['run_index']:04d}_last_five.json"
        )
        run_path.write_text(
            json.dumps(run_export, indent=2),
            encoding="utf-8",
        )
        combined_export.append(run_export)

    (output_directory / "last_five_sequences.json").write_text(
        json.dumps(combined_export, indent=2),
        encoding="utf-8",
    )

    with (output_directory / "last_five_sequences.pkl").open("wb") as handle:
        pickle.dump(combined_export, handle, protocol=pickle.HIGHEST_PROTOCOL)

    csv_path = output_directory / "last_five_sequences.csv"
    fieldnames = [
        "run_index",
        "seed",
        "position_among_last_five",
        "time_seconds",
        "score",
        "isomorphic",
        "ged",
        "sequence",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_rows)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["mnist", "cifar10"], default="cifar10")
    parser.add_argument("--results-root", default="paper_results")
    parser.add_argument("--output-dir", default=os.path.join("paper_results", "DGD_reconstruction"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--time-budget", type=float, default=5 * 60)
    parser.add_argument("--n-runs", type=int, default=20)
    parser.add_argument("--n-points", type=int, default=500)
    parser.add_argument("--level", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--reward-label", default="Gram_error_fro")
    parser.add_argument("--prior-simulations", type=int, default=300)
    parser.add_argument("--max-workers", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    observations = load_observations_history(
        dataset_name=args.dataset,
        results_root=args.results_root,
    )

    print(f"Observations loaded for dataset '{args.dataset}'.")
    X_history = observations["X_history"]
    n_nodes = observations["n_nodes"]

    target_graph = nx.gnp_random_graph(
        n_nodes,
        observations["edge_probability"],
        seed=observations["graph_seed"],
    )
    synchron = True

    if not nx.is_connected(target_graph):
        raise ValueError(
            "The target graph reconstructed from metadata is not connected. "
            "Use the exact graph-generation procedure from the training script."
        )

    output_directory = Path(
        os.path.join(args.output_dir, args.dataset)
        or Path(args.results_root) / args.dataset / "reconstruction"
    )
    output_directory.mkdir(parents=True, exist_ok=True)

    infected_graph, initial_graph, attackers_set, N_A, prior = (
        prepare_reconstruction(
            target_graph=target_graph,
            X_history=X_history,
            synchron=synchron,
            attacker_node=observations["attacker_node"],
            observed_neighbor_ids=observations["neighbor_node_ids"],
            args=args,
        )
    )
    print(f"Prior computed for {len(prior)} candidate moves.", flush=True)

    start = time.time()
    results = run_parallel_reconstructions(
        target_graph=infected_graph,
        initial_graph=initial_graph,
        synchron=synchron,
        X_history=X_history,
        attackers_set=attackers_set,
        N_A=N_A,
        prior=prior,
        args=args,
    )

    with (output_directory / "reconstruction_runs.pkl").open("wb") as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Export the last five incumbent-best sequences and their associated logs.
    export_last_five_sequences(results, output_directory)

    output_directory = Path(args.output_dir) / args.dataset

    with (output_directory / "reconstruction_runs.pkl").open("rb") as handle:
        results = pickle.load(handle)

    time_grid, score_matrix = plot_reward_evolution(
        results=results,
        t_min=0.1,
        t_max=args.time_budget,
        n_points=args.n_points,
        output_path=output_directory / "DGD_reward_evolution.png",
        confidence_level=0.95,
    )

    np.savez_compressed(
        output_directory / "reward_evolution.npz",
        time_grid=time_grid,
        score_matrix=score_matrix,
    )

    summary_rows = []
    for result in results:
        final_score = (
            result["history"][-1][1]
            if result["history"]
            else float("nan")
        )
        summary_rows.append({
            "run_index": result["run_index"],
            "seed": result["seed"],
            "final_score": final_score,
            "isomorphic": result["isomorphic"],
            "ged": result["ged"],
            "runtime_seconds": result["runtime_seconds"],
            "number_of_improvements": result["number_of_improvements"],
            "exported_last_sequences": len(result["last_five_improvements"]),
        })

    with (output_directory / "summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)

    with (output_directory / "target_graph.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(graph_to_json(infected_graph, n_nodes), handle, indent=4)

    metadata = {
        "dataset": args.dataset,
        "n_nodes": n_nodes,
        "n_edges": target_graph.number_of_edges(),
        "attacker": observations["attacker_node"],
        "neighbors": observations["neighbor_node_ids"].tolist(),
        "time_step": len(X_history),
        "n_runs": args.n_runs,
        "time_budget_seconds": args.time_budget,
        "reward_label": args.reward_label,
        "last_sequences_exported_per_run": 5,
        "isomorphism_rate": float(np.mean([r["isomorphic"] for r in results])),
        "total_runtime_seconds": time.time() - start,
    }
    with (output_directory / "metadata.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(metadata, handle, indent=4)

    print(
        f"Completed {args.n_runs} runs in {(time.time() - start) / 60:.2f} min. "
        f"Results: {output_directory.resolve()}",
        flush=True,
    )

if __name__ == "__main__":
    main()
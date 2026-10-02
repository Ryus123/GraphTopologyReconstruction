#############################################################
#### Import
#############################################################
import argparse
import os
import pickle
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations
from multiprocessing import cpu_count
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from graph_topology_attacks.utils import signal_propagation, setting_attacks, guess_graph_initialization, load_model
from graph_topology_attacks.algo.offline_prior import prior_function, adapt, playout


start_time = 0.0
time_budget = 0.0
global_best_score = float("-inf")
score_history = []

#############################################################
#### Functions
#############################################################

def NRPA(G: nx.Graph, X_history: np.ndarray, n_edges: int, attackers_set: list, 
         N_A: int, time_step: int, level: int, N: int, policy: dict, prior: dict, 
         reward_label: str, ):
    global start_time, time_budget, global_best_score, score_history

    if time.time() - start_time >= time_budget:
        return global_best_score, [], score_history

    if level == 0:
        score, sequence = playout(
            policy,
            prior,
            G,
            X_history,
            attackers_set,
            N_A,
            time_step,
            reward_label,
            n_edges,
            tau=1.0,
        )

        if score > global_best_score:
            global_best_score = score
            score_history.append((time.time() - start_time, global_best_score))

        return score, sequence, score_history

    local_best_score = float("-inf")
    best_seq = []

    for _ in range(N):
        if time.time() - start_time >= time_budget:
            break

        reward, sequence, _ = NRPA(G.copy(), X_history, n_edges, attackers_set, N_A, 
                                   time_step, level - 1, N, policy, prior, reward_label, )

        if reward == 1.0:
            if reward > global_best_score:
                global_best_score = reward
                score_history.append((time.time() - start_time, reward))
            return reward, sequence, score_history

        if reward >= local_best_score:
            local_best_score = reward
            best_seq = sequence

        policy = adapt(policy, prior, best_seq, G, X_history, attackers_set, 
                       n_edges, N_A, time_step, reward_label, tau=1.0, alpha=0.8, )

    return local_best_score, best_seq, score_history


def single_NRPA_run(arguments):
    global start_time
    global time_budget
    global global_best_score
    global score_history

    (
        run_index,
        seed,
        G_bis,
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
    ) = arguments

    random.seed(seed)
    np.random.seed(seed)

    start_time = time.time()
    time_budget = budget
    global_best_score = float("-inf")
    score_history = []

    _, best_sequence, history = NRPA(G_bis.copy(), X_history, number_of_edges, attackers_set, 
                                     N_A, time_step, level, N, {}, prior, reward_label, )

    reconstructed_graph = G_bis.copy()
    reconstructed_graph.add_edges_from(best_sequence)

    return {
        "run_index": run_index,
        "seed": seed,
        "history": list(history),
        "best_sequence": best_sequence,
        "reconstructed_graph": reconstructed_graph,
    }


def run(number_of_nodes: int, number_of_edges: int, X_history: np.ndarray, attackers_set: list, 
        N_A: int, init_attackers: dict, time_step: int, level: int, N: int, reward_label: str, 
        time_budget: float = 500.0, N_runs: int = 1, G_target: nx.Graph = None, base_seed: int = 42, 
        max_workers: int = None, ):

    model = load_model(
        n_nodes=number_of_nodes,
        r_label=reward_label,
        instance="ER",
    )

    G_initial = guess_graph_initialization(
        number_of_nodes,
        init_attackers,
        G_target,
    )

    moves_to_compute = [
        sorted((node_a, node_b))
        for node_a, node_b in combinations(
            G_initial.nodes(),
            2,
        )
        if not str(node_a).startswith("attacker")
        and not str(node_b).startswith("attacker")
        and not G_initial.has_edge(node_a, node_b)
    ]

    prior = prior_function(
        moves_to_compute,
        model,
        G_initial,
        X_history,
        number_of_edges,
        N_A,
        attackers_set,
        metric=reward_label,
        nb_simu=300,
        limited_model=True,
    )

    parameters = [
        (
            run_index,
            base_seed + run_index,
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
            time_budget,
        )
        for run_index in range(N_runs)
    ]

    if max_workers is None:
        allocated_cpus = int(
            os.environ.get(
                "SLURM_CPUS_PER_TASK",
                cpu_count(),
            )
        )

        workers = min(
            N_runs,
            max(1, allocated_cpus),
        )
    else:
        workers = min(
            N_runs,
            max(1, max_workers),
        )

    print(
        f"Running {N_runs} NRPA runs "
        f"with {workers} workers.",
        flush=True,
    )

    run_results = []

    with ProcessPoolExecutor(
        max_workers=workers
    ) as executor:
        futures = {
            executor.submit(
                single_NRPA_run,
                parameters_run,
            ): parameters_run[0]
            for parameters_run in parameters
        }

        for future in as_completed(futures):
            run_index = futures[future]

            try:
                result = future.result()
                run_results.append(result)

                final_score = (
                    result["history"][-1][1]
                    if result["history"]
                    else np.nan
                )

                print(
                    f"Run {run_index + 1}/{N_runs}: "
                    f"score={final_score}",
                    flush=True,
                )

            except Exception as error:
                raise RuntimeError(
                    f"NRPA run {run_index} failed: {error!r}"
                ) from error

    run_results.sort(
        key=lambda result: result["run_index"]
    )

    score_history_per_run = [
        [
            result["history"],
            result["best_sequence"],
        ]
        for result in run_results
    ]

    reconstructed_graphs = [
        result["reconstructed_graph"]
        for result in run_results
    ]

    return (
        score_history_per_run,
        reconstructed_graphs,
        G_initial,
    )


def build_log_time_score_matrix(
    score_history_per_run,
    t_min=0.1,
    t_max=600.0,
    n_points=500,
    initial_score=0.0,
):
    if not 0 < t_min < t_max:
        raise ValueError("Expected 0 < t_min < t_max.")

    time_grid = np.geomspace(t_min, t_max, n_points)
    matrix = np.full(
        (len(score_history_per_run), n_points), initial_score, dtype=float
    )
    best_sequences = []

    for run_index, result in enumerate(score_history_per_run):
        if not result or len(result) < 2:
            continue

        history, best_sequence = result[0], result[1]
        best_sequences.append(best_sequence)
        history = sorted(history, key=lambda item: item[0])

        current_score = initial_score
        history_index = 0

        for time_index, current_time in enumerate(time_grid):
            while (
                history_index < len(history)
                and history[history_index][0] <= current_time
            ):
                current_score = history[history_index][1]
                history_index += 1

            matrix[run_index, time_index] = current_score

    return time_grid, matrix, best_sequences


def isomorphism_test(sequences, target_G, init_G_guess):
    results = []
    for sequence in sequences:
        reconstructed = init_G_guess.copy()
        reconstructed.add_edges_from(sequence)
        results.append(nx.is_isomorphic(reconstructed, target_G))
    return results


def group_name(group, node_names=None):
    return "_".join(
        str(node_names.get(node, node)) if node_names else str(node)
        for node in group
    )


def mean_metric(metric, group):
    return round(float(np.mean([metric[node] for node in group])), 4)


def plot_results_all_nodes(
    results_per_groups,
    target_G,
    t_min=0.1,
    t_max=600.0,
    n_points=500,
    initial_score=0.0,
    centrality_type="degree_avg",
    node_names=None,
    fname="Score_evolution.png",
):
    if not results_per_groups:
        print("No results to plot. Exiting.")
        return
    groups_to_plot = sorted(results_per_groups.keys())
    with plt.rc_context({
        "font.family": "serif",
        "font.serif": [
            "Times New Roman",
            "Times",
            "Nimbus Roman",
            "DejaVu Serif",
        ],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "lines.linewidth": 1.8,
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }):
        color_map = plt.colormaps["tab10"]
        # line_styles = ["-", "--", "-.", ":"]
        line_styles = ["-"]
        figure, axis = plt.subplots(figsize=(3.5, 2.8))
        minimum_reward = 1.0
        valid_groups = []
        for index, group in enumerate(groups_to_plot):
            (
                score_path,
                isomorphism_rate,
                median_ged,
            ) = results_per_groups[group]
            if not os.path.isfile(score_path):
                print(
                    f"Warning: file not found for "
                    f"{group}: {score_path}"
                )
                continue
            with open(score_path, "rb") as file:
                score_history_per_run = pickle.load(file)
            time_grid, score_matrix, _ = build_log_time_score_matrix(
                score_history_per_run,
                t_min=t_min,
                t_max=t_max,
                n_points=n_points,
                initial_score=initial_score,
            )
            if score_matrix.shape[0] == 0:
                continue
            median_reward = np.median(
                score_matrix,
                axis=0,
            )
            first_quartile = np.quantile(
                score_matrix,
                0.25,
                axis=0,
            )
            third_quartile = np.quantile(
                score_matrix,
                0.75,
                axis=0,
            )
            minimum_reward = min(
                minimum_reward,
                float(np.min(first_quartile)),
            )
            color = color_map( (4+ index) % color_map.N)
            group_label = group_name(
                group,
                node_names,
            )
            axis.fill_between(
                time_grid,
                first_quartile,
                third_quartile,
                color=color,
                alpha=0.18,
                linewidth=0,
                zorder=1,
            )
            axis.plot(
                time_grid,
                median_reward,
                color=color,
                linestyle=line_styles[
                    index % len(line_styles)
                ],
                linewidth=2.0,
                label=(
                    f"{group_label.replace('_', ' & ')}: median, "
                    f"iso. {100 * isomorphism_rate:.0f}%"
                ),
                zorder=3,
            )
            valid_groups.append(group)
        if not valid_groups:
            plt.close(figure)
            print("No valid result was found. Exiting.")
            return
        axis.axhline(
            1.0,
            color="#4A4A4A",
            linestyle=":",
            linewidth=1.2,
            label="Target reward",
            zorder=2,
        )
        axis.set_xscale("log")
        axis.set_xlim(t_min, t_max)
        reward_margin = max(
            0.001,
            0.08 * (1.0 - minimum_reward),
        )
        axis.set_ylim(
            0.97,
            1.001,
        )
        axis.set_xlabel(
            "Wall-clock time (s)",
            labelpad=8,
        )
        axis.set_ylabel(
            "Best reconstruction reward",
            labelpad=8,
        )
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
        axis.tick_params(
            axis="both",
            which="both",
            color="#707070",
            pad=5,
        )
        axis.spines["left"].set_color("#707070")
        axis.spines["bottom"].set_color("#707070")
        number_of_entries = len(valid_groups) + 1
        legend_columns = min(
            1,
            number_of_entries,
        )
        legend_rows = int(
            np.ceil(
                number_of_entries
                / legend_columns
            )
        )
        axis.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.20),
            ncol=legend_columns,
            frameon=False,
            handlelength=2.4,
            handletextpad=0.6,
            columnspacing=1.2,
        )
        bottom_margin = (
            0.23
            + 0.05 * max(
                0,
                legend_rows - 1,
            )
        )
        figure.subplots_adjust(
            left=0.13,
            right=0.98,
            top=0.97,
            bottom=bottom_margin,
        )
        output_path = Path(fname)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        figure.savefig(
            output_path,
            dpi=600,
            bbox_inches="tight",
            pad_inches=0.05,
            facecolor="white",
        )
        figure.savefig(
            output_path.with_suffix(".pdf"),
            bbox_inches="tight",
            pad_inches=0.05,
            facecolor="white",
        )
        plt.close(figure)
    print(
        f"Plots saved as: {output_path} and "
        f"{output_path.with_suffix('.pdf')}"
    )


def process_configuration(
    attacker_group,
    target_graph,
    n_nodes,
    time_step,
    level,
    N,
    reward_label,
    time_budget,
    n_runs,
    node_names=None,
    output_directory=os.path.join("paper_results","florentine_families"),
    base_seed=42,
):
    attacker_group = tuple(attacker_group)
    print(f"Processing configuration: {attacker_group}")
    config_start_time = time.time()

    # Distinct but reproducible seed for each combination.
    seed = base_seed + sum((index + 1) * int(node) for index, node in enumerate(attacker_group))
    random.seed(seed)
    np.random.seed(seed)

    attacker_mapping = {
        node: f"attacker_{node}"
        for node in attacker_group
    }
    print(f"Attacker mapping: {attacker_mapping}")
    G_infected, attackers_neighbors = setting_attacks(
        target_graph,
        attacker_mapping,
    )

    init_attackers = {
        attacker: list(nx.neighbors(G_infected, attacker))
        for attacker in attacker_mapping.values()
    }
    print(f"Initial attackers and their neighbors: {init_attackers}")
    N_A = sum(len(neighbors) for neighbors in attackers_neighbors.values())
    attackers_set = list(attackers_neighbors.keys())

    X_history = signal_propagation(
        G_infected,
        attackers_set,
        N_A,
        time_step,
    )

    ( score_history_per_run, reconstructed_graphs, initial_graph,) = run(
        number_of_nodes=n_nodes,
        number_of_edges=target_graph.number_of_edges(),
        X_history=X_history,
        attackers_set=attackers_set,
        N_A=N_A,
        init_attackers=init_attackers,
        time_step=time_step,
        level=level,
        N=N,
        reward_label=reward_label,
        time_budget=time_budget,
        N_runs=n_runs,
        G_target=G_infected,
        base_seed=seed,
        max_workers=cpu_count() - 4,
    )


    os.makedirs(output_directory, exist_ok=True)
    identifier = group_name(attacker_group, node_names)
    score_path = os.path.join(
        output_directory,
        f"score_history_{len(attacker_group)}_attackers_{identifier}.pkl",
    )

    with open(score_path, "wb") as file:
        pickle.dump(score_history_per_run, file)

    elapsed_minutes = (time.time() - config_start_time) / 60

    isomorphism_results = [
    nx.is_isomorphic(
            reconstructed_graph,
            G_infected,)
        for reconstructed_graph in reconstructed_graphs
    ]

    ged_results = [
        nx.graph_edit_distance(
            reconstructed_graph,
            G_infected,
            timeout=5.0,)
            for reconstructed_graph in reconstructed_graphs
    ]

    valid_ged_results = [ float(ged) for ged in ged_results if ged is not None ]

    result = {
        "attacker_ids": attacker_group,
        "attacker_names": identifier,
        "n_attackers": len(attacker_group),
        "n_runs": n_runs,
        "score_path": score_path,
        "elapsed_minutes": elapsed_minutes,
        "isomorphism_rate": float(
            np.mean(isomorphism_results)
        ),
        "median_GED": (
            float(np.median(valid_ged_results))
            if valid_ged_results
            else np.nan
        ),
        "seed": seed,
    }
    print(result, flush=True)
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-attackers", type=int, default=3)
    parser.add_argument(
        "--n-configurations",
        type=int,
        default=20,
        help="Number of combinations to sample. Use 0 for all combinations.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--time-budget", type=float, default=3600)
    parser.add_argument("--n-runs", type=int, default=50)
    parser.add_argument("--n-points", type=int, default=900)
    parser.add_argument("--level", type=int, default=18)
    parser.add_argument("--iterations", type=int, default=7)
    parser.add_argument("--reward-label", default="Gram_error_fro")
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--output-dir", default=os.path.join("paper_results","florentine_families"))
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    target_graph = nx.convert_node_labels_to_integers(
        nx.florentine_families_graph(),
        first_label=0,
        ordering="default",
        label_attribute="original_label",
    )
    
    node_names = nx.get_node_attributes(target_graph, "original_label")
    n_nodes = target_graph.number_of_nodes()

    if not 1 <= args.n_attackers <= n_nodes:
        raise ValueError(
            f"n_attackers must be between 1 and {n_nodes}; got {args.n_attackers}."
        )

    # {1: 'Medici',3: 'Peruzzi', 12: 'Guadagni'}
    selected_groups = [(1,12), (1,3,12)]

    print(f"Selected {len(selected_groups)} combinations with 2 and 3 attackers.")

    max_workers = 1

    print(f"Using {max_workers} worker process(es).")
    time_step = n_nodes * 2
    start = time.time()
    results_per_groups = {}

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_configuration,
                group,
                target_graph,
                n_nodes,
                time_step,
                args.level,
                args.iterations,
                args.reward_label,
                args.time_budget,
                args.n_runs,
                node_names,
                args.output_dir,
                args.seed,
            ): group
            for group in selected_groups
        }

        for future in as_completed(futures):
            group = futures[future]
            try:
                result = future.result()
                results_per_groups[group] = (
                    result["score_path"],
                    result["isomorphism_rate"],
                    result["median_GED"],
                )
            except Exception as error:
                print(f"Error for configuration {group}: {error!r}", flush=True)


    os.makedirs(args.output_dir, exist_ok=True)
    for n_attackers in [2, 3]: 
        filtered_results = { group: value for group, value in results_per_groups.items() if len(group) == n_attackers } 
        results_path = os.path.join( args.output_dir, f"results_{n_attackers}_attackers.pkl", ) 
        with open(results_path, "wb") as file: 
            pickle.dump(filtered_results, file)

    print(f"Results saved as: {results_path}")

    if not args.no_plot:
        plot_results_all_nodes(
            results_per_groups=results_per_groups, 
            target_G=target_graph, 
            t_min=0.1, 
            t_max=3600, 
            n_points=900, 
            initial_score=0.0, 
            node_names=node_names,
            fname=os.path.join(
                "paper_figures",
                "florentine_families",
                f"reward_evolution_per_N_attackers.png",
            ),
        )

    print(f"Total runtime: {(time.time() - start) / 60:.2f} minutes")


if __name__ == "__main__":
    main()

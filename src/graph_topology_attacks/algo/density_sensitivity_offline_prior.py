#############################################################
#### Import
#############################################################
import os
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

from graph_topology_attacks.utils import signal_propagation, setting_attacks, load_model, guess_graph_initialization
from graph_topology_attacks.algo.offline_prior import prior_function, get_legal_moves, play

import time
from multiprocessing import Pool, cpu_count
from itertools import combinations
import pandas as pd

##############################################################
#### Functions
##############################################################
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
        weights_vect = []
        actions_vect = []

        for m in legal_moves:
            key_child = key + "," + str(m)
            p_m = policy.get(key_child, 0.0)
            ex = np.exp((p_m/tau) + prior[str(sorted(m))])
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


def adapt(policy, prior, sequence, G:nx.Graph, X_history:np.array, 
          attackers_set:list, nb_edges:int, N_A:int, time_step:int, 
          reward_label:str, tau=1.0, alpha=1.0):
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
            policy[key_child] = (policy.get(key_child, 0.0)/tau) + prior[str(sorted(m))]
            z += np.exp(policy[key_child])

        for m in moves:
            key_child = key + "," + str(m)
            polp[key_child] = polp.get(key_child, 0.0)
            p_m = policy[key_child]
            polp[key_child] -= (alpha/tau) * ((np.exp(p_m) / z) - int(sorted(m) == sorted(action)))

        G_bis, *_ = play(G_bis, action[0], action[1], X_history, attackers_set, N_A, time_step, reward_label)
        moves.remove(action)
        key += "," + str(action)

    return polp


def NRPA(G: nx.Graph, X_history: np.array, n_edges: int, 
         attackers_set: list, N_A: int, time_step: int, 
         level: int, N: int, policy: dict, prior: dict, reward_label: str):

    global start_time, time_budget, global_best_score, score_history

    if time.time() - start_time >= time_budget:
        return global_best_score, [], score_history
    
    if level == 0:
        score, sequence = playout(policy, prior, G.copy(), X_history, 
                                  attackers_set, N_A, time_step, reward_label, 
                                  n_edges, tau=1.0)

        curr_time = time.time()

        if score > global_best_score:
            global_best_score = score
            score_history.append((curr_time - start_time, global_best_score))

        return score, sequence, score_history

    else:
        local_best_score = float("-inf")
        best_seq = []

        for _ in range(N):
            if time.time() - start_time >= time_budget:
                break

            reward, sequence, _ = NRPA(G.copy(), X_history, 
                                       n_edges, attackers_set, N_A, time_step, 
                                       level - 1, N, policy, prior, reward_label)

            if reward == 1.0:
                if reward > global_best_score:
                    global_best_score = reward
                    score_history.append((time.time() - start_time, reward))
                    
                return reward, sequence, score_history

            if reward > local_best_score:
                local_best_score = reward
                best_seq = sequence

                policy = adapt(policy, prior, sequence, G, X_history, 
                               attackers_set, n_edges, N_A, time_step, 
                               reward_label, tau=1.0, alpha=1.1)

    return local_best_score, best_seq, score_history


def single_NRPA_run(args):
    global start_time, time_budget, global_best_score, score_history

    (G_bis, X_history, number_of_edges, attackers_set, N_A, time_step, 
     level, N, prior, reward_label, budget,
     p) = args

    start_time = time.time()
    time_budget = budget
    global_best_score = float("-inf")
    score_history = []

    policy = {}

    _, best_seq, history = NRPA(G_bis.copy(), X_history, number_of_edges, 
                                attackers_set, N_A, time_step, level, N,
                                policy, prior, reward_label)
    
    return [history[-1], best_seq, p]



def run(number_of_nodes: int, number_of_edges: int, X_history: np.array, 
        attackers_set: list, N_A: int, init_attackers: dict, time_step: int, 
        level: int, N: int, reward_label: str, time_budget: float = 500.0,
        p=1/3):

    model = load_model(n_nodes=number_of_nodes, 
                    r_label=reward_label, 
                    instance="ER")

    G = guess_graph_initialization(number_of_nodes,init_attackers)
    G_bis = G.copy()

    moves_to_compute = [
        sorted((a, b)) for 
        a, b in combinations(G_bis.nodes(), 2) if 
        not (
            str(a).startswith("attacker") or 
            str(b).startswith("attacker")
            ) 
        and not G_bis.has_edge(a, b)
    ]

    prior = prior_function(moves_to_compute, model, G_bis, X_history, 
                           number_of_edges, N_A, attackers_set, 
                           metric=reward_label)

    params = (G_bis, X_history, number_of_edges, attackers_set, 
                  N_A, time_step, level, N, prior, reward_label, 
                  time_budget,
                  p)
        
    score_history_per_run = single_NRPA_run(params)

    best_seq = score_history_per_run[1]
    G_guess = G_bis.copy()
    G_guess.add_edges_from(best_seq)

    return score_history_per_run, G_guess


def build_points_data(score_history_per_run, time_budget):
    data = {"runtime": [], "p": [], "reward": [], "iso_test": []}

    for ele, iso_test in score_history_per_run:
        last_improvment, _, p = ele
        r = last_improvment[1]
        data["runtime"].append(last_improvment[0] if r == 1.0 else time_budget)
        data["reward"].append(r)
        data["p"].append(p)
        data["iso_test"].append(iso_test)
    
    return data

def solve_one_problem(args):

    (
        target_graph,
        p,
        attackers_values,
        n_nodes,
        time_step,
        level,
        N,
        reward_label,
        TIME_BUDGET,
    ) = args

    for n_att in attackers_values:

        a_mapping = {i: f"attacker_{i}" for i in range(n_att)}

        G_infected, attackers_neighbors = setting_attacks(
            target_graph,
            a_mapping
        )

        init_attackers = {
            a: list(nx.neighbors(G_infected, a))
            for a in a_mapping.values()
        }

        N_A = sum(len(neigh) for neigh in attackers_neighbors.values())
        attackers_set = list(attackers_neighbors.keys())

        X_history = signal_propagation(
            G_infected,
            attackers_set,
            N_A,
            time_step
        )

        score_history_per_run, G_guess = run(
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
            time_budget=TIME_BUDGET,
            p=p
        )

    return score_history_per_run, nx.is_isomorphic(G_guess, G_infected)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import bootstrap

SUCCESS_COLOR = "#2C6E9B"
TIMEOUT_COLOR = "#555555"
TIMEOUT_RTOL = 1e-9
TIMEOUT_ATOL = 1e-6
RANDOM_SEED = 0


def aggregate_runtime_vs_p(data, time_budget):
    dataframe = pd.DataFrame(data).copy()

    required_columns = {
        "p",
        "reward",
        "runtime",
        "iso_test",
    }
    missing_columns = required_columns.difference(
        dataframe.columns
    )
    if missing_columns:
        raise ValueError(
            "Missing columns: "
            + ", ".join(sorted(missing_columns))
        )

    for column in [
        "p",
        "reward",
        "runtime",
    ]:
        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

    dataframe = dataframe.dropna(
        subset=[
            "p",
            "reward",
            "runtime",
        ]
    ).copy()

    if dataframe.empty:
        raise ValueError(
            "No valid observation remains after "
            "numeric conversion."
        )

    if (dataframe["runtime"] < 0).any():
        raise ValueError(
            "Runtime values must be non-negative."
        )

    dataframe["runtime_raw"] = dataframe["runtime"]
    dataframe["runtime"] = np.minimum(
        dataframe["runtime_raw"],
        time_budget,
    )

    dataframe["is_timeout"] = (
        np.isclose(
            dataframe["runtime_raw"],
            time_budget,
            rtol=TIMEOUT_RTOL,
            atol=TIMEOUT_ATOL,
        )
        | (dataframe["runtime_raw"] >= time_budget)
    )

    # Same success definition as in the replot script.
    dataframe["is_success"] = np.isclose(
        dataframe["reward"],
        1.0,
        rtol=1e-9,
        atol=1e-12,
    )

    rows = []

    for p_value, group in dataframe.groupby(
        "p",
        sort=True,
    ):
        number_of_runs = len(group)
        number_of_successes = int(
            group["is_success"].sum()
        )
        number_of_timeouts = int(
            group["is_timeout"].sum()
        )
        success_rate = (
            number_of_successes / number_of_runs
        )

        runtime_values = group["runtime"].to_numpy(
            dtype=float
        )
        mean_runtime = float(np.mean(runtime_values))

        if number_of_runs == 1:
            confidence_interval_low = mean_runtime
            confidence_interval_high = mean_runtime
        else:
            bootstrap_result = bootstrap(
                data=(runtime_values,),
                statistic=np.mean,
                confidence_level=0.95,
                n_resamples=5000,
                method="percentile",
                random_state=RANDOM_SEED,
            )
            confidence_interval_low = float(
                bootstrap_result.confidence_interval.low
            )
            confidence_interval_high = float(
                bootstrap_result.confidence_interval.high
            )

        rows.append({
            "p": float(p_value),
            "n_total": number_of_runs,
            "n_success": number_of_successes,
            "n_timeouts": number_of_timeouts,
            "success_rate": success_rate,
            "mean_runtime_all": mean_runtime,
            "ci_lo": confidence_interval_low,
            "ci_hi": confidence_interval_high,
        })

    aggregated_dataframe = (
        pd.DataFrame(rows)
        .sort_values("p")
        .reset_index(drop=True)
    )

    return dataframe, aggregated_dataframe


def plot_runtime_vs_p(
    data,
    time_budget,
    fname="runtime_vs_p.png",
):
    dataframe, aggregated = aggregate_runtime_vs_p(
        data,
        time_budget,
    )

    style = {
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
        "axes.titlesize": 10,
        "axes.titleweight": "semibold",
        "legend.fontsize": 9,
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
        "lines.linewidth": 1.5,
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }

    with plt.rc_context(style):
        figure, (
            success_axis,
            runtime_axis,
        ) = plt.subplots(
            nrows=2,
            ncols=1,
            figsize=(3.5, 2.8),
            sharex=True,
        )

        figure.subplots_adjust(
            left=0.12,
            right=0.97,
            top=0.94,
            bottom=0.13,
            hspace=0.24,
        )

        # ----------------------------------------------------------
        # Panel (a): success rate
        # ----------------------------------------------------------
        success_axis.plot(
            aggregated["p"],
            aggregated["success_rate"],
            color="black",
            linewidth=1.4,
            marker="o",
            markersize=4.0,
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=0.9,
            zorder=3,
        )

        success_axis.fill_between(
            aggregated["p"],
            0.0,
            aggregated["success_rate"],
            color="#D9D9D9",
            alpha=0.35,
            linewidth=0,
            zorder=1,
        )

        success_axis.set_ylabel(
            "Success rate",
            labelpad=18,
        )
        success_axis.set_ylim(
            0.0,
            1.02,
        )
        success_axis.set_yticks([
            0.0,
            0.25,
            0.5,
            0.75,
            1.0,
        ])

        success_axis.text(
            -0.2,
            1.075,
            "(a)",
            transform=success_axis.transAxes,
            fontsize=10,
            fontweight="bold",
            horizontalalignment="left",
            verticalalignment="top",
            clip_on=False,
        )

        # ----------------------------------------------------------
        # Panel (b): mean runtime including timeouts
        # ----------------------------------------------------------
        runtime_data = aggregated.dropna(
            subset=[
                "mean_runtime_all",
                "ci_lo",
                "ci_hi",
            ]
        ).copy()

        if not runtime_data.empty:
            runtime_axis.fill_between(
                runtime_data["p"],
                runtime_data["ci_lo"],
                runtime_data["ci_hi"],
                color=SUCCESS_COLOR,
                alpha=0.16,
                linewidth=0,
                zorder=1,
            )

            runtime_axis.plot(
                runtime_data["p"],
                runtime_data["mean_runtime_all"],
                color=SUCCESS_COLOR,
                linewidth=1.4,
                marker="o",
                markersize=4.0,
                markerfacecolor="white",
                markeredgecolor=SUCCESS_COLOR,
                markeredgewidth=0.9,
                zorder=3,
            )

        runtime_axis.axhline(
            time_budget,
            color=TIMEOUT_COLOR,
            linestyle=(0, (4, 3)),
            linewidth=0.8,
            alpha=0.75,
            zorder=0,
        )

        for row in aggregated.itertuples(index=False):
            if row.n_timeouts == 0:
                continue

            runtime_axis.scatter(
                row.p,
                time_budget,
                marker="x",
                s=20,
                color=TIMEOUT_COLOR,
                linewidths=1.0,
                zorder=5,
                clip_on=False,
            )

            runtime_axis.annotate(
                str(row.n_timeouts),
                xy=(
                    row.p,
                    time_budget * 0.9,
                ),
                xytext=(0, 9),
                textcoords="offset points",
                fontsize=7,
                color=TIMEOUT_COLOR,
                horizontalalignment="center",
                verticalalignment="bottom",
                annotation_clip=False,
            )

        runtime_axis.annotate(
            "Timeout",
            xy=(
                1.05,
                time_budget * 0.8,
            ),
            xycoords=(
                "axes fraction",
                "data",
            ),
            xytext=(0, 3),
            textcoords="offset points",
            fontsize=7,
            color="#444444",
            horizontalalignment="right",
            verticalalignment="bottom",
        )

        runtime_axis.set_xlabel(
            r"ER edge probability $p$",
            labelpad=18,
        )
        runtime_axis.set_ylabel(
            "Mean runtime (s)",
            labelpad=18,
        )

        upper_values = np.concatenate([
            (
                runtime_data["ci_hi"].to_numpy(
                    dtype=float
                )
                if not runtime_data.empty
                else np.array([], dtype=float)
            ),
            np.array(
                [time_budget],
                dtype=float,
            ),
        ])
        runtime_upper_limit = float(
            np.nanmax(upper_values)
        )

        runtime_axis.set_ylim(
            bottom=0.0,
            top=max(
                time_budget * 1.15,
                runtime_upper_limit * 1.10,
            ),
        )

        runtime_axis.text(
            -0.2,
            1.075,
            "(b)",
            transform=runtime_axis.transAxes,
            fontsize=10,
            fontweight="bold",
            horizontalalignment="left",
            verticalalignment="top",
            clip_on=False,
        )

        # ----------------------------------------------------------
        # Common formatting
        # ----------------------------------------------------------
        for axis in (
            success_axis,
            runtime_axis,
        ):
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.spines["left"].set_linewidth(0.8)
            axis.spines["bottom"].set_linewidth(0.8)

            axis.tick_params(
                axis="both",
                which="major",
                width=0.8,
                length=3,
                direction="out",
                pad=8,
            )

            axis.grid(
                visible=True,
                axis="y",
                color="#B0B0B0",
                linestyle="-",
                linewidth=0.5,
                alpha=0.25,
                zorder=0,
            )
            axis.set_axisbelow(True)

        output_path = Path(fname)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_base = output_path.with_suffix("")
        pdf_path = output_base.with_suffix(".pdf")
        png_path = output_base.with_suffix(".png")

        figure.savefig(
            pdf_path,
            bbox_inches="tight",
            pad_inches=0.08,
            facecolor="white",
        )
        figure.savefig(
            png_path,
            dpi=600,
            bbox_inches="tight",
            pad_inches=0.08,
            facecolor="white",
        )
        plt.close(figure)

    aggregated_path = output_base.parent / (
        output_base.name + "_aggregated.csv"
    )
    aggregated.to_csv(
        aggregated_path,
        index=False,
    )

    print(
        f"Figures saved as: {pdf_path} and {png_path}"
    )
    print(
        f"Aggregated results saved as: "
        f"{aggregated_path}"
    )

    return dataframe, aggregated


if __name__ == "__main__":

    TIME_BUDGET = 3600.0

    n_nodes = 10
    time_step = n_nodes * 2
    level = 20
    N = 1000
    nb_simu = 30
    reward_label = "Gram_error_fro"

    attackers_values = [1]

    problem_to_solve = []
    list_p = [p for p in np.linspace(0.23, 1, num=10)]
    for p in list_p:

        for _ in range(nb_simu):
            target_graph = nx.gnp_random_graph(n_nodes, p)
            attempts = 0
            while not nx.is_connected(target_graph):
                attempts += 1
                if attempts > 30:
                    raise ValueError("Target graph must be connected")
                target_graph = nx.gnp_random_graph(n_nodes, p)
            
            problem_to_solve.append((target_graph, p))

    t_start = time.time()
    tasks = [
        (target_graph, p, attackers_values, n_nodes, 
         time_step, level, N, reward_label, TIME_BUDGET)
        for (target_graph, p) in problem_to_solve
        ]

    with Pool(processes=cpu_count() - 4) as pool:
        results_list = pool.map(solve_one_problem, tasks)

    data = build_points_data(results_list, time_budget=TIME_BUDGET)

    dataframe, aggregated = plot_runtime_vs_p(
        data=data, time_budget=TIME_BUDGET, 
        fname=os.path.join( "paper_figures","runtime_vs_p", "runtime_vs_p",),)

    print(f"Total execution time: {(time.time() - t_start)/60:.2f} minutes")
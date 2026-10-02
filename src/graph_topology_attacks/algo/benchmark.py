#############################################################
#### Import
#############################################################
import copy
import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from multiprocessing import Pool, cpu_count

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import networkx as nx
import numpy as np

from pathlib import Path
from statistics import NormalDist
from matplotlib.lines import Line2D

from graph_topology_attacks.utils import signal_propagation, setting_attacks, construct_graph_from_vne_instance
from graph_topology_attacks.metrics import Laplacian_Spectral_Distance, Spectral_Radius_Distance
from graph_topology_attacks.algo.nrpa_without_init import run as run_nrpa_without_init
from graph_topology_attacks.algo.offline_prior import run as run_prior_function

EXCLUDED_INSTANCES = {"events.txt", "node_names.json", "test_network", "stats.json"}
#############################################################
#### functions
#############################################################
def parameter_key(params):
    return params["instance"], params["algo"], tuple(params["algo_params"])


def stats_path_for(params):
    algo_params = params["algo_params"]
    reward_label = algo_params[-1]
    return os.path.join(
        "paper_results", "benchmark",
        f"{params['instance']}_benchmark_{params['algo']}_"
        f"{algo_params[0]}_{algo_params[1]}_{reward_label}_"
        f"model_{params['instance']}.json",
    )


def get_instance_paths(instance_repo):
    if instance_repo == "Garr201201":
        data_repo = os.path.join(
            "src", "graph_topology_attacks", "data",
            "vne_instances", instance_repo,
        )
    else:
        data_repo = os.path.join(
            "src", "graph_topology_attacks", "data", instance_repo,
        )

    paths = []
    for filename in os.listdir(data_repo):
        if filename in EXCLUDED_INSTANCES:
            continue

        keep = filename.startswith(("PP", "slice"))
        if not keep:
            parts = filename.split("_")
            try:
                keep = len(parts) > 1 and int(parts[1]) < 30
            except ValueError:
                keep = False

        if keep:
            paths.append(os.path.join(data_repo, filename))

    paths.sort()
    if not paths:
        raise ValueError(f"Aucune instance trouvee dans {data_repo}")
    return paths


def export_records(records, filename):
    records_copy = copy.deepcopy(records)
    for algo_stats in records_copy.values():
        for pb_size_stats in algo_stats.values():
            for metric_key, metric_values in pb_size_stats.items():
                if isinstance(metric_values, np.ndarray):
                    pb_size_stats[metric_key] = metric_values.tolist()
                elif isinstance(metric_values, np.generic):
                    pb_size_stats[metric_key] = metric_values.item()

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(records_copy, file, indent=4)


def parallel_run(task):
    """Execute une combinaison (jeu de parametres, instance) dans un worker."""
    params, instance_path = task
    instance_repo = params["instance"]

    model_instance = {
        "Garr201201": "ER", "ER": "ER", "large_ER": "ER",
        "medium_ER_25": "ER", "Barabasi_Albert": "BA",
        "peertube_follow": "ER",
    }

    digest = hashlib.sha256(instance_path.encode("utf-8")).digest()
    seed = 42 + int.from_bytes(digest[:4], "little") % 100000
    random.seed(seed)
    np.random.seed(seed)

    target_graph = construct_graph_from_vne_instance(instance_path)
    n_nodes = target_graph.number_of_nodes()
    n_edges = target_graph.number_of_edges()

    a_mapping = {0: "attacker_0"}
    time_step = min(n_nodes * 2, 24)

    G_infected, attackers_neighbors = setting_attacks(target_graph, a_mapping)
    N_A = sum(len(neighbors) for neighbors in attackers_neighbors.values())
    attackers_set = list(attackers_neighbors)
    X_history = signal_propagation(
        G_infected, attackers_set, N_A, time_step
    )
    init_attackers = {
        attacker: list(G_infected.neighbors(attacker))
        for attacker in a_mapping.values()
    }

    algo = params["algo"]
    algo_params = params["algo_params"]

    if algo == "nrpa_without_init":
        level, N, budget, reward_label = algo_params
        G_guess, reward, error, time_taken = run_nrpa_without_init(
            n_nodes, n_edges, X_history, attackers_set, N_A,
            init_attackers, time_step, level=level, N=N,
            reward_label=reward_label, budget=budget,
        )
    elif algo == "offline_prior":
        level, N, budget, reward_label = algo_params
        G_guess, reward, error, time_taken = run_prior_function(
            n_nodes, n_edges, X_history, attackers_set, N_A,
            init_attackers, time_step, level=level, N=N,
            reward_label=reward_label, budget=budget,
            instance_name=model_instance[instance_repo],
        )
    else:
        raise ValueError(f"Algorithme inconnu : {algo}")

    return (
        parameter_key(params), G_infected, attackers_set, G_guess,
        reward, error, time_taken, reward_label,
    )


def empty_metrics():
    return {
        "success_rate": np.array([]),
        "local_degree_finded": np.array([]),
        "average_score": np.array([]),
        "average_error": np.array([]),
        "average_GED": np.array([]),
        "wl_test_rate": np.array([]),
        "average_TV_distance": np.array([]),
        "average_time": np.array([]),
    }


def degree_distribution(graph):
    count = Counter(degree for _, degree in graph.degree())
    total = sum(count.values())
    return {degree: number / total for degree, number in count.items()}


def build_stats(params, results):
    """Agrege les resultats d'un jeu de parametres apres le Pool global."""
    stats_record = {}

    for result in results:
        G_infected, attackers_set, G_guess, reward, error, time_taken, reward_label = result
        n_nodes = G_infected.number_of_nodes()
        success = int(nx.is_isomorphic(G_infected, G_guess))
        time_s = float(str(time_taken).split()[0])

        target_neighbors = {
            neighbor
            for attacker in attackers_set
            for neighbor in G_infected.neighbors(attacker)
        }
        guess_neighbors = {
            neighbor
            for attacker in attackers_set
            for neighbor in G_guess.neighbors(attacker)
        }
        target_degrees = sorted(degree for _, degree in G_infected.degree(target_neighbors))
        guess_degrees = sorted(degree for _, degree in G_guess.degree(guess_neighbors))
        local_degree_found = int(target_degrees == guess_degrees)

        dist_target = degree_distribution(G_infected)
        dist_guess = degree_distribution(G_guess)
        all_degrees = set(dist_target) | set(dist_guess)
        tv_distance = 0.5 * sum(
            abs(dist_target.get(d, 0) - dist_guess.get(d, 0))
            for d in all_degrees
        )
        wl_test = int(
            nx.weisfeiler_lehman_graph_hash(G_infected)
            == nx.weisfeiler_lehman_graph_hash(G_guess)
        )

        record_key = f"{params['algo']},{reward_label}"
        stats_record.setdefault(record_key, {})
        metrics = stats_record[record_key].setdefault(str(n_nodes), empty_metrics())

        metrics["success_rate"] = np.append(metrics["success_rate"], success)
        metrics["local_degree_finded"] = np.append(
            metrics["local_degree_finded"], local_degree_found
        )
        metrics["average_score"] = np.append(metrics["average_score"], reward[-1])
        metrics["average_error"] = np.append(metrics["average_error"], error[-1])
        metrics["average_GED"] = np.append(
            metrics["average_GED"],
            nx.graph_edit_distance(G_infected, G_guess, timeout=5),
        )
        metrics["average_time"] = np.append(metrics["average_time"], time_s)
        metrics["average_TV_distance"] = np.append(
            metrics["average_TV_distance"], tv_distance
        )
        metrics["wl_test_rate"] = np.append(metrics["wl_test_rate"], wl_test)
        Laplacian_Spectral_Distance(G_infected, G_guess, records=metrics)
        Spectral_Radius_Distance(G_infected, G_guess, records=metrics)

    export_records(stats_record, stats_path_for(params))
    print(f"Benchmark completed for : {params}")
    return stats_record


ALGORITHMS = {
    "NRPA": {
        "internal_name": "nrpa_without_init",
        "color": "#0072B2",
        "marker": "o",
    },
    "GNRPA with Offline prior": {
        "internal_name": "offline_prior",
        "color": "#D55E00",
        "marker": "s",
    },
}

TIMEOUTS = {
    "ER": 30_000.0,
    "medium_ER_25": 30_000.0,
    "Barabasi_Albert": 15_000.0,
    "peertube_follow": 30_000.0,
    "Garr201201": 15_000.0,
}

CONFIDENCE_LEVEL = 0.95
TIMEOUT_RTOL = 1e-6
TIMEOUT_ATOL = 1e-6
RANDOM_SEED = 42
SHOW_INDIVIDUAL_RUNS = False
MAX_VISIBLE_POINTS = 100
ALGORITHM_OFFSETS = [-0.18, 0.18]
BOX_WIDTH = 0.29

FIGSIZE_MAIN = (9, 3)
FIGSIZE_SECONDARY = (8.2, 6.0)


def normalize_benchmark_results(stats_record):
    normalized_results = {}

    for display_name, configuration in ALGORITHMS.items():
        internal_name = configuration["internal_name"]

        matching_keys = [
            key
            for key in stats_record
            if key.split(",", 1)[0].strip() == internal_name
        ]

        merged_sizes = defaultdict(
            lambda: defaultdict(list)
        )

        for key in matching_keys:
            algorithm_results = stats_record[key]

            for size, metrics in algorithm_results.items():
                for metric, raw_values in metrics.items():
                    values = np.asarray(
                        raw_values,
                        dtype=float,
                    ).reshape(-1)

                    values = values[
                        np.isfinite(values)
                    ]

                    merged_sizes[str(size)][metric].extend(
                        values.tolist()
                    )

        normalized_results[display_name] = {
            size: dict(metrics)
            for size, metrics in merged_sizes.items()
        }

    return normalized_results


def get_common_sizes(all_results):
    return sorted({
        int(size)
        for algorithm_results in all_results.values()
        for size in algorithm_results
    })

def get_values(
    all_results,
    algorithm_name,
    metric,
    size,
    allow_missing=False,
):
    size_key = str(size)
    algorithm_results = all_results[algorithm_name]

    if size_key not in algorithm_results:
        if allow_missing:
            return np.empty(0, dtype=float)
        raise KeyError(
            f"Size n={size} is missing for "
            f"{algorithm_name}."
        )

    size_results = algorithm_results[size_key]

    if metric not in size_results:
        if allow_missing:
            return np.empty(0, dtype=float)
        available_metrics = sorted(size_results)
        raise KeyError(
            f"Metric '{metric}' is missing for "
            f"{algorithm_name}, n={size}. "
            f"Available metrics: {available_metrics}"
        )

    values = np.asarray(
        size_results[metric],
        dtype=float,
    ).reshape(-1)

    return values[np.isfinite(values)]


def is_binary(values, tolerance=1e-10):
    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[np.isfinite(values)]

    return bool(
        values.size
        and np.all(
            np.isclose(
                values,
                0.0,
                atol=tolerance,
                rtol=0.0,
            )
            | np.isclose(
                values,
                1.0,
                atol=tolerance,
                rtol=0.0,
            )
        )
    )


def wilson_confidence_interval(
    successes,
    trials,
    confidence_level=CONFIDENCE_LEVEL,
):
    if trials <= 0:
        return np.nan, np.nan, np.nan

    z_value = NormalDist().inv_cdf(
        1.0 - (1.0 - confidence_level) / 2.0
    )

    proportion = successes / trials
    denominator = 1.0 + z_value**2 / trials

    center = (
        proportion
        + z_value**2 / (2.0 * trials)
    ) / denominator

    half_width = (
        z_value
        * np.sqrt(
            proportion * (1.0 - proportion) / trials
            + z_value**2 / (4.0 * trials**2)
        )
        / denominator
    )

    return (
        proportion,
        max(0.0, center - half_width),
        min(1.0, center + half_width),
    )


def base_positions(sizes):
    return np.arange(
        len(sizes),
        dtype=float,
    )


def configure_x_axis(axis, sizes):
    positions = base_positions(sizes)

    axis.set_xticks(positions)
    axis.set_xticklabels(
        [str(size) for size in sizes]
    )

    axis.set_xlim(
        -0.60,
        len(sizes) - 0.40,
    )


def configure_common_axis(axis, ylabel):
    axis.set_ylabel(
        ylabel,
        labelpad=4,
    )

    axis.grid(
        axis="y",
        color="#D9D9D9",
        linewidth=0.55,
        alpha=0.70,
    )

    axis.tick_params(
        axis="both",
        pad=2,
    )


def add_panel_label(axis, label):
    axis.text(
        -0.13,
        1.04,
        label,
        transform=axis.transAxes,
        fontsize=9,
        fontweight="bold",
        horizontalalignment="left",
        verticalalignment="bottom",
    )


def show_missing_panel(
    axis,
    metric,
    panel_label,
):
    axis.text(
        0.5,
        0.5,
        f"Missing metric\n{metric}",
        transform=axis.transAxes,
        horizontalalignment="center",
        verticalalignment="center",
        fontsize=8,
        color="#555555",
    )

    add_panel_label(
        axis,
        panel_label,
    )

    axis.set_xticks([])
    axis.set_yticks([])

    for spine in axis.spines.values():
        spine.set_visible(False)


def plot_binary_comparison(
    axis,
    all_results,
    sizes,
    metric,
    ylabel,
    panel_label,
):
    positions = base_positions(sizes)
    plotted = False

    for algorithm_name, configuration in ALGORITHMS.items():
        x_values = []
        proportions = []
        lower_bounds = []
        upper_bounds = []

        for position, size in zip(
            positions,
            sizes,
        ):
            values = get_values(
                all_results,
                algorithm_name,
                metric,
                size,
                allow_missing=True,
            )

            if not values.size:
                continue

            if not is_binary(values):
                raise ValueError(
                    f"Metric '{metric}' is not binary for "
                    f"{algorithm_name}, n={size}."
                )

            binary_values = np.isclose(
                values,
                1.0,
                atol=1e-10,
                rtol=0.0,
            ).astype(int)

            proportion, lower, upper = (
                wilson_confidence_interval(
                    int(binary_values.sum()),
                    binary_values.size,
                )
            )

            x_values.append(position)
            proportions.append(proportion)
            lower_bounds.append(lower)
            upper_bounds.append(upper)

        if not x_values:
            continue

        plotted = True

        x_values = np.asarray(x_values)
        proportions = np.asarray(proportions)
        lower_bounds = np.asarray(lower_bounds)
        upper_bounds = np.asarray(upper_bounds)

        axis.plot(
            x_values,
            proportions,
            color=configuration["color"],
            marker=configuration["marker"],
            markersize=4.2,
            markerfacecolor="white",
            markeredgewidth=1.1,
            linewidth=1.5,
            zorder=3,
        )

        axis.errorbar(
            x_values,
            proportions,
            yerr=np.vstack([
                proportions - lower_bounds,
                upper_bounds - proportions,
            ]),
            fmt="none",
            color=configuration["color"],
            capsize=2.2,
            elinewidth=0.9,
            capthick=0.9,
            zorder=2,
        )

    if not plotted:
        show_missing_panel(
            axis,
            metric,
            panel_label,
        )
        return

    configure_x_axis(
        axis,
        sizes,
    )

    configure_common_axis(
        axis,
        ylabel,
    )

    axis.set_ylim(
        -0.04,
        1.04,
    )

    axis.set_yticks([
        0.0,
        0.25,
        0.5,
        0.75,
        1.0,
    ])

    add_panel_label(
        axis,
        panel_label,
    )


def draw_single_boxplot(
    axis,
    values,
    position,
    color,
    width=BOX_WIDTH,
):
    axis.boxplot(
        [values],
        positions=[position],
        widths=width,
        patch_artist=True,
        showfliers=False,
        whis=(5, 95),
        manage_ticks=False,
        zorder=2,
        boxprops={
            "facecolor": color,
            "edgecolor": color,
            "linewidth": 0.9,
            "alpha": 0.22,
        },
        medianprops={
            "color": color,
            "linewidth": 1.6,
        },
        whiskerprops={
            "color": color,
            "linewidth": 0.8,
            "alpha": 0.90,
        },
        capprops={
            "color": color,
            "linewidth": 0.8,
            "alpha": 0.90,
        },
    )


def draw_individual_points(
    axis,
    values,
    position,
    color,
    random_generator,
):
    if (
        not SHOW_INDIVIDUAL_RUNS
        or not values.size
    ):
        return

    if values.size > MAX_VISIBLE_POINTS:
        visible_values = random_generator.choice(
            values,
            MAX_VISIBLE_POINTS,
            replace=False,
        )
    else:
        visible_values = values

    jitter = random_generator.uniform(
        -BOX_WIDTH * 0.28,
        BOX_WIDTH * 0.28,
        visible_values.size,
    )

    axis.scatter(
        np.full(
            visible_values.size,
            position,
        ) + jitter,
        visible_values,
        s=4.5,
        color=color,
        alpha=0.18,
        edgecolors="none",
        rasterized=True,
        zorder=1,
    )


def plot_continuous_comparison(
    axis,
    all_results,
    sizes,
    metric,
    ylabel,
    panel_label,
    log_scale=False,
):
    positions = base_positions(sizes)
    plotted = False

    for algorithm_index, (
        algorithm_name,
        configuration,
    ) in enumerate(ALGORITHMS.items()):
        offset = ALGORITHM_OFFSETS[
            algorithm_index
        ]

        random_generator = np.random.default_rng(
            RANDOM_SEED
            + sum(ord(character) for character in metric)
            + algorithm_index * 10_000
        )

        median_x = []
        median_y = []

        for position, size in zip(
            positions,
            sizes,
        ):
            values = get_values(
                all_results,
                algorithm_name,
                metric,
                size,
                allow_missing=True,
            )

            if not values.size:
                continue

            plotted = True
            shifted_position = position + offset

            draw_single_boxplot(
                axis,
                values,
                shifted_position,
                configuration["color"],
            )

            draw_individual_points(
                axis,
                values,
                shifted_position,
                configuration["color"],
                random_generator,
            )

            median_x.append(
                shifted_position
            )

            median_y.append(
                float(np.median(values))
            )

        if median_x:
            axis.plot(
                median_x,
                median_y,
                color=configuration["color"],
                linewidth=1.0,
                marker=configuration["marker"],
                markersize=3.4,
                markerfacecolor="white",
                markeredgewidth=0.9,
                zorder=3,
            )

    if not plotted:
        show_missing_panel(
            axis,
            metric,
            panel_label,
        )
        return

    configure_x_axis(
        axis,
        sizes,
    )

    configure_common_axis(
        axis,
        ylabel,
    )

    if log_scale:
        axis.set_yscale("log")

    add_panel_label(
        axis,
        panel_label,
    )


def split_runtime_values(
    values,
    timeout,
):
    values = np.asarray(
        values,
        dtype=float,
    )

    timeout_mask = (
        np.isclose(
            values,
            timeout,
            rtol=TIMEOUT_RTOL,
            atol=TIMEOUT_ATOL,
        )
        | (values >= timeout)
    )

    return (
        values[~timeout_mask],
        values[timeout_mask],
    )


def plot_runtime_comparison(
    axis,
    all_results,
    sizes,
    timeout,
    metric="average_time",
    ylabel="Runtime (s)",
    panel_label="(b)",
):
    positions = base_positions(sizes)
    plotted = False

    for algorithm_index, (
        algorithm_name,
        configuration,
    ) in enumerate(ALGORITHMS.items()):
        offset = ALGORITHM_OFFSETS[
            algorithm_index
        ]

        random_generator = np.random.default_rng(
            RANDOM_SEED
            + 20_000
            + algorithm_index
        )

        median_x = []
        median_y = []

        for position, size in zip(
            positions,
            sizes,
        ):
            values = get_values(
                all_results,
                algorithm_name,
                metric,
                size,
                allow_missing=True,
            )

            if not values.size:
                continue

            plotted = True
            shifted_position = position + offset
            displayed_values = np.minimum(
                values,
                timeout,
            )

            draw_single_boxplot(
                axis,
                displayed_values,
                shifted_position,
                configuration["color"],
            )

            completed_values, timed_out_values = (
                split_runtime_values(
                    values,
                    timeout,
                )
            )

            draw_individual_points(
                axis,
                completed_values,
                shifted_position,
                configuration["color"],
                random_generator,
            )

            if timed_out_values.size:
                axis.scatter(
                    shifted_position,
                    timeout,
                    marker="x",
                    s=18,
                    color=configuration["color"],
                    linewidths=1.0,
                    zorder=5,
                )

                axis.annotate(
                    str(timed_out_values.size),
                    xy=(
                        shifted_position,
                        timeout,
                    ),
                    xytext=(0, 7),
                    textcoords="offset points",
                    fontsize=5.8,
                    color=configuration["color"],
                    horizontalalignment="center",
                    verticalalignment="top",
                )

            median_x.append(
                shifted_position
            )

            median_y.append(
                float(
                    np.median(
                        displayed_values
                    )
                )
            )

        if median_x:
            axis.plot(
                median_x,
                median_y,
                color=configuration["color"],
                linewidth=1.0,
                marker=configuration["marker"],
                markersize=3.4,
                markerfacecolor="white",
                markeredgewidth=0.9,
                zorder=3,
            )

    if not plotted:
        show_missing_panel(
            axis,
            metric,
            panel_label,
        )
        return

    axis.axhline(
        timeout,
        color="#555555",
        linestyle=(0, (4, 3)),
        linewidth=0.8,
        alpha=0.75,
        zorder=0,
    )

    axis.annotate(
        "Timeout",
        xy=(0.1, timeout),
        xycoords=(
            "axes fraction",
            "data",
        ),
        xytext=(17, 0),
        textcoords="offset points",
        fontsize=6.5,
        color="#444444",
        horizontalalignment="right",
        verticalalignment="bottom",
    )

    configure_x_axis(
        axis,
        sizes,
    )

    configure_common_axis(
        axis,
        ylabel,
    )

    bottom_limit, top_limit = axis.get_ylim()

    axis.set_ylim(
        min(0.0, bottom_limit),
        max(top_limit, timeout * 1.1),
    )

    add_panel_label(
        axis,
        panel_label,
    )


def metric_is_binary_for_all_available_data(
    all_results,
    sizes,
    metric,
):
    found = False

    for algorithm_name in ALGORITHMS:
        for size in sizes:
            values = get_values(
                all_results,
                algorithm_name,
                metric,
                size,
                allow_missing=True,
            )

            if not values.size:
                continue

            found = True

            if not is_binary(values):
                return False

    return found


def plot_auto_comparison(
    axis,
    all_results,
    sizes,
    metric,
    ylabel,
    panel_label,
):
    is_binary_metric = (
        metric_is_binary_for_all_available_data(
            all_results,
            sizes,
            metric,
        )
    )

    if is_binary_metric:
        plot_binary_comparison(
            axis,
            all_results,
            sizes,
            metric,
            ylabel,
            panel_label,
        )
    else:
        plot_continuous_comparison(
            axis,
            all_results,
            sizes,
            metric,
            ylabel,
            panel_label,
        )


def add_algorithm_legend(figure):
    handles = [
        Line2D(
            [0],
            [0],
            color=configuration["color"],
            marker=configuration["marker"],
            markerfacecolor="white",
            markeredgecolor=configuration["color"],
            markersize=4.2,
            linewidth=1.4,
            label=algorithm_name,
        )
        for algorithm_name, configuration
        in ALGORITHMS.items()
    ]

    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=len(handles),
        frameon=False,
        handlelength=2.0,
        columnspacing=2.0,
    )


#############################################################
#### Plot
#############################################################
def plot_benchmark_results(
    stats_record,
    instance_repo,
    output_directory="paper_figures/benchmark",
):
    all_results = normalize_benchmark_results(
        stats_record
    )

    sizes = get_common_sizes(
        all_results
    )

    if not sizes:
        print(
            f"No result available for {instance_repo}."
        )
        return

    output_directory = Path(
        output_directory
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    timeout = TIMEOUTS.get(
        instance_repo,
        30_000.0,
    )

    style = {
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
        "legend.fontsize": 11,
        "lines.linewidth": 1.5,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    }

    with plt.rc_context(style):
        figure, axes = plt.subplots(
            1,
            3,
            figsize=FIGSIZE_MAIN,
            sharex=True,
        )

        plot_binary_comparison(
            axes[0],
            all_results,
            sizes,
            metric="success_rate",
            ylabel="Success rate",
            panel_label="(a)",
        )

        plot_runtime_comparison(
            axes[1],
            all_results,
            sizes,
            timeout=timeout,
            metric="average_time",
            ylabel="Runtime (s)",
            panel_label="(b)",
        )

        plot_continuous_comparison(
            axes[2],
            all_results,
            sizes,
            metric="average_GED",
            ylabel="GED",
            panel_label="(c)",
        )

        figure.supxlabel(
            "Number of nodes, $n$"
        )

        add_algorithm_legend(
            figure
        )

        figure.subplots_adjust(
            left=0.07,
            right=0.99,
            bottom=0.16,
            top=0.82,
            wspace=0.30,
        )

        main_pdf_path = (
            output_directory
            / f"{instance_repo}_algorithm_comparison_main.pdf"
        )

        main_png_path = (
            output_directory
            / f"{instance_repo}_algorithm_comparison_main.png"
        )

        figure.savefig(
            main_pdf_path,
            facecolor="white",
        )

        figure.savefig(
            main_png_path,
            dpi=600,
            facecolor="white",
        )

        plt.close(figure)

        figure, axes = plt.subplots(
            2,
            2,
            figsize=FIGSIZE_SECONDARY,
            sharex=True,
        )

        plot_auto_comparison(
            axes[0, 0],
            all_results,
            sizes,
            metric="wl_test_rate",
            ylabel="WL test rate",
            panel_label="(a)",
        )

        plot_continuous_comparison(
            axes[0, 1],
            all_results,
            sizes,
            metric="average_TV_distance",
            ylabel="Total variation distance",
            panel_label="(b)",
        )

        plot_continuous_comparison(
            axes[1, 0],
            all_results,
            sizes,
            metric="laplacian_spectral_distance",
            ylabel="Laplacian spectral distance",
            panel_label="(c)",
        )

        plot_auto_comparison(
            axes[1, 1],
            all_results,
            sizes,
            metric="local_degree_finded",
            ylabel="Local degree recovery rate",
            panel_label="(d)",
        )

        axes[1, 0].set_xlabel(
            "Number of nodes, $n$"
        )

        axes[1, 1].set_xlabel(
            "Number of nodes, $n$"
        )

        add_algorithm_legend(
            figure
        )

        figure.subplots_adjust(
            left=0.10,
            right=0.98,
            bottom=0.09,
            top=0.91,
            wspace=0.30,
            hspace=0.22,
        )

        secondary_pdf_path = (
            output_directory
            / f"{instance_repo}_algorithm_comparison_secondary.pdf"
        )

        secondary_png_path = (
            output_directory
            / f"{instance_repo}_algorithm_comparison_secondary.png"
        )

        figure.savefig(
            secondary_pdf_path,
            facecolor="white",
        )

        figure.savefig(
            secondary_png_path,
            dpi=600,
            facecolor="white",
        )

        plt.close(figure)

    print(
        f"Main figure saved as: {main_pdf_path} "
        f"and {main_png_path}",
        flush=True,
    )

    print(
        f"Secondary figure saved as: "
        f"{secondary_pdf_path} and "
        f"{secondary_png_path}",
        flush=True,
    )


#############################################################
#### Main
#############################################################
def main():
    random.seed(42)
    np.random.seed(42)

    parameters = [
        {"instance": "ER", "time_step": "default", "algo": "nrpa_without_init", "algo_params": (20, 1000, 30000.0, "Gram_error_fro")},
        {"instance": "ER", "time_step": "default", "algo": "offline_prior", "algo_params": (20, 1000, 30000.0, "Gram_error_fro")},

        {"instance": "Barabasi_Albert", "time_step": "default", "algo": "nrpa_without_init", "algo_params": (20, 1000, 30000.0, "Gram_error_fro")},
        {"instance": "Barabasi_Albert", "time_step": "default", "algo": "offline_prior", "algo_params": (20, 1000, 30000.0, "Gram_error_fro")},

        {"instance": "medium_ER_25", "time_step": "default", "algo": "nrpa_without_init", "algo_params": (20, 5000, 30000.0, "Gram_error_fro")},
        {"instance": "medium_ER_25", "time_step": "default", "algo": "offline_prior", "algo_params": (20, 5000, 30000.0, "Gram_error_fro")},

        {"instance": "peertube_follow", "time_step": "default", "algo": "nrpa_without_init", "algo_params": (20, 5000, 30000.0, "Gram_error_fro")},
        {"instance": "peertube_follow", "time_step": "default", "algo": "offline_prior", "algo_params": (20, 5000, 30000.0, "Gram_error_fro")},

        {"instance": "Garr201201", "time_step": "default", "algo": "nrpa_without_init", "algo_params": (20, 1000, 30000.0, "Gram_error_fro")},
        {"instance": "Garr201201", "time_step": "default", "algo": "offline_prior", "algo_params": (20, 1000, 30000.0, "Gram_error_fro")},
    ]

    loaded_or_computed = {}
    pending_params = {}
    all_tasks = []

    for params in parameters:
        key = parameter_key(params)
        path = stats_path_for(params)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as file:
                loaded_or_computed[key] = json.load(file)
            print(f"Loaded existing results : {path}")
        else:
            pending_params[key] = params
            all_tasks.extend(
                (params, instance_path)
                for instance_path in get_instance_paths(params["instance"])
            )

    grouped_results = defaultdict(list)
    if all_tasks:
        workers = max(1, min(len(all_tasks), cpu_count() - 4))
        print(f"Launching {len(all_tasks)} tasks in a global Pool of {workers} workers")

        with Pool(processes=workers) as pool:
            for completed, output in enumerate(
                pool.imap_unordered(parallel_run, all_tasks, chunksize=1), start=1):
                key, *result = output
                grouped_results[key].append(tuple(result))
                print(f"Completed : {completed}/{len(all_tasks)}", flush=True)

    for key, params in pending_params.items():
        loaded_or_computed[key] = build_stats(params, grouped_results[key])

    merged_by_instance = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(list)
            )
        )
    )

    for key, stats in loaded_or_computed.items():
        instance_repo = key[0]

        plot_group = (
            "ER"
            if instance_repo in {"ER", "medium_ER_25"}
            else instance_repo
        )

        for algorithm_key, size_results in stats.items():
            for size, metrics in size_results.items():
                for metric, values in metrics.items():
                    values = np.asarray(
                        values,
                        dtype=float,
                    ).reshape(-1)

                    values = values[
                        np.isfinite(values)
                    ]

                    merged_by_instance[
                        plot_group
                    ][algorithm_key][str(size)][metric].extend(
                        values.tolist()
                    )

    for instance_repo, merged_stats in merged_by_instance.items():
        plot_benchmark_results(
            stats_record=merged_stats,
            instance_repo=instance_repo,
            output_directory=os.path.join(
                "paper_figures",
                "benchmark",
            ),
        )


if __name__ == "__main__":
    main()

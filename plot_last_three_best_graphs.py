#############################################################
#### Import
#############################################################
import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx

from graph_topology_attacks.utils import setting_attacks, guess_graph_initialization

#############################################################
#### Config
#############################################################
plt.rcParams.update(
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
                "legend.fontsize": 11,
                "lines.linewidth": 1.5,
                "axes.linewidth": 0.8,
                "xtick.major.width": 0.8,
                "ytick.major.width": 0.8,
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
    }
)

#############################################################
#### functions
#############################################################
def load_experiment(
    dataset,
    results_root,
    reconstruction_dir=None,
):
    dataset_dir = Path(results_root) / dataset
    metadata_path = dataset_dir / "metadata.json"

    reconstruction_dir = (
        Path(reconstruction_dir)
        if reconstruction_dir is not None
        else dataset_dir / "reconstruction"
    )

    sequences_path = (
        reconstruction_dir / "last_five_sequences.json"
    )

    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Missing metadata: {metadata_path}"
        )

    if not sequences_path.is_file():
        raise FileNotFoundError(
            f"Missing sequences: {sequences_path}"
        )

    with metadata_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        metadata = json.load(handle)

    with sequences_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        runs = json.load(handle)

    required = {"nodes", "attacker"}
    missing = required.difference(metadata)

    if missing:
        raise KeyError(
            f"Missing metadata fields: {sorted(missing)}"
        )

    return metadata, runs, reconstruction_dir


def build_graphs(metadata):
    n_nodes = int(metadata["nodes"])
    edge_probability = float(
        metadata.get("edge_probability", 1.0 / 3.0)
    )
    graph_seed = int(
        metadata.get("graph_seed", 4332)
    )
    attacker_node = int(metadata["attacker"])

    target_graph = nx.gnp_random_graph(
        n_nodes,
        edge_probability,
        seed=graph_seed,
    )

    if not nx.is_connected(target_graph):
        raise ValueError(
            "The graph regenerated from metadata is "
            "disconnected. Use the exact graph-generation "
            "procedure of the training run."
        )

    attacker_label = f"attacker_{attacker_node}"

    infected_target, _ = setting_attacks(
        target_graph,
        {attacker_node: attacker_label},
    )

    init_attackers = {
        attacker_label: list(
            infected_target.neighbors(attacker_label)
        )
    }

    initial_graph = guess_graph_initialization(
        n_nodes,
        init_attackers,
        infected_target,
    )

    return (
        infected_target,
        initial_graph,
        attacker_label,
    )


def decode_node(node):
    if isinstance(node, int):
        return node

    if (
        isinstance(node, str)
        and node.lstrip("-").isdigit()
    ):
        return int(node)

    return node


def decode_sequence(sequence):
    decoded = []

    for edge in sequence:
        if len(edge) != 2:
            raise ValueError(f"Invalid edge: {edge}")

        decoded.append(
            (
                decode_node(edge[0]),
                decode_node(edge[1]),
            )
        )

    return decoded


def canonical_edge(edge):
    return frozenset(edge)


def format_node_label(node):
    if isinstance(node, str) and node.startswith("attacker_"):
        attacker_id = node.removeprefix("attacker_")
        return f"A{attacker_id}"

    return str(node)


def draw_graph_panel(
    axis,
    graph,
    positions,
    attacker_label,
    initial_edge_set,
    target_edge_set,
    title,
    is_target=False,
):
    if is_target:
        known_edges = list(graph.edges())
        correct_added_edges = []
        incorrect_added_edges = []
    else:
        known_edges = [
            edge
            for edge in graph.edges()
            if canonical_edge(edge) in initial_edge_set
        ]

        correct_added_edges = [
            edge
            for edge in graph.edges()
            if (
                canonical_edge(edge)
                not in initial_edge_set
                and canonical_edge(edge)
                in target_edge_set
            )
        ]

        incorrect_added_edges = [
            edge
            for edge in graph.edges()
            if (
                canonical_edge(edge)
                not in initial_edge_set
                and canonical_edge(edge)
                not in target_edge_set
            )
        ]

    node_colors = [
        "#B2182B"
        if node == attacker_label
        else "#E7EEF8"
        for node in graph.nodes()
    ]

    label_colors = {
        node: (
            "#FFFFFF"
            if node == attacker_label
            else "#1F2937"
        )
        for node in graph.nodes()
    }

    node_sizes = [
        700
        if node == attacker_label
        else 500
        for node in graph.nodes()
    ]

    nx.draw_networkx_nodes(
        graph,
        positions,
        ax=axis,
        node_size=node_sizes,
        node_color=node_colors,
        edgecolors="#26354A",
        linewidths=1.25,
    )

    nx.draw_networkx_edges(
        graph,
        positions,
        ax=axis,
        edgelist=known_edges,
        edge_color=(
            "#98A2B3"
            if not is_target
            else "#26354A"
        ),
        width=1.8,
        alpha=0.95,
    )

    if correct_added_edges:
        nx.draw_networkx_edges(
            graph,
            positions,
            ax=axis,
            edgelist=correct_added_edges,
            edge_color="#16875D",
            width=1.8,
            alpha=1.0,
        )

    if incorrect_added_edges:
        nx.draw_networkx_edges(
            graph,
            positions,
            ax=axis,
            edgelist=incorrect_added_edges,
            edge_color="#C94747",
            width=1.8,
            style="dashed",
            alpha=1.0,
        )

    for node, (
        x_coordinate,
        y_coordinate,
    ) in positions.items():
        axis.text(
            x_coordinate,
            y_coordinate,
            format_node_label(node),
            color=label_colors[node],
            fontweight="bold",
            ha="center",
            va="center",
            zorder=5,
        )

    axis.set_title(
        title,
        pad=4,
        linespacing=1.35,
    )

    axis.margins(0.12)
    axis.set_axis_off()


def node_role(
    graph,
    node,
    attacker_label,
):
    if node == attacker_label:
        return "attacker"

    if graph.has_edge(node, attacker_label):
        return "attacker_neighbor"

    return "other"


def constrained_isomorphic_relabel(
    candidate_graph,
    target_graph,
    attacker_label,
):

    candidate = candidate_graph.copy()
    target = target_graph.copy()

    for node in candidate.nodes():
        candidate.nodes[node]["role"] = node_role(
            candidate,
            node,
            attacker_label,
        )

    for node in target.nodes():
        target.nodes[node]["role"] = node_role(
            target,
            node,
            attacker_label,
        )

    node_match = (
        nx.algorithms.isomorphism
        .categorical_node_match(
            "role",
            None,
        )
    )

    matcher = nx.algorithms.isomorphism.GraphMatcher(
        candidate,
        target,
        node_match=node_match,
    )

    for mapping in matcher.isomorphisms_iter():
        if mapping.get(attacker_label) != attacker_label:
            continue

        relabeled_graph = nx.relabel_nodes(
            candidate_graph,
            mapping,
            copy=True,
        )

        return relabeled_graph, mapping

    return candidate_graph.copy(), None


def relabel_edge_set(
    edge_set,
    mapping,
):
    if mapping is None:
        return edge_set.copy()

    relabeled_edges = set()

    for edge in edge_set:
        endpoints = tuple(edge)

        if len(endpoints) != 2:
            raise ValueError(
                f"Invalid canonical edge: {edge}"
            )

        source, target = endpoints

        relabeled_edges.add(
            canonical_edge(
                (
                    mapping.get(source, source),
                    mapping.get(target, target),
                )
            )
        )

    return relabeled_edges


def plot_run(
    run,
    initial_graph,
    target_graph,
    attacker_label,
    output_directory,
    layout_seed=42,
):
    improvements = run.get(
        "last_five_improvements",
        [],
    )[-3:]

    seed_index = int(run["run_index"])

    estimation_labels = [
        "Best estimate (t−2)",
        "Best estimate (t−1)",
        "Final estimate",
        ]
    
    if not improvements:
        print(
            f"Seed {seed_index}: no improvement to plot."
        )
        return

    positions = nx.spring_layout(
        target_graph,
        seed=layout_seed,
        k=1.25 / max(
            target_graph.number_of_nodes() ** 0.5,
            1.0,
        ),
        iterations=300,
    )

    initial_edge_set = {
        canonical_edge(edge)
        for edge in initial_graph.edges()
    }

    target_edge_set = {
        canonical_edge(edge)
        for edge in target_graph.edges()
    }

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(7.0, 5.6),
        squeeze=False,
    )

    flat_axes = axes.ravel()

    for improvement_index, (
        axis,
        improvement,
    ) in enumerate(
        zip(
            flat_axes[:3],
            improvements,
        ),
        start=1,
    ):
        sequence = decode_sequence(
            improvement["sequence"]
        )

        candidate = initial_graph.copy()
        candidate.add_edges_from(sequence)

        aligned_candidate, mapping = (
            constrained_isomorphic_relabel(
                candidate_graph=candidate,
                target_graph=target_graph,
                attacker_label=attacker_label,
            )
        )

        constrained_isomorphic = (
            mapping is not None
        )

        aligned_initial_edge_set = relabel_edge_set(
            initial_edge_set,
            mapping,
        )

        score = float(improvement["score"])
        elapsed = float(
            improvement["time_seconds"]
        )
        ged = improvement.get("ged")

        ged_text = (
            "N/A"
            if ged is None
            else f"{float(ged):.1f}"
        )

        title = (
            f"{estimation_labels[improvement_index - 1]}\n"
            f"Reward = {score:.5f}, "
            f"Time = {elapsed:.2f} s\n"
        )

        draw_graph_panel(
            axis=axis,
            graph=aligned_candidate,
            positions=positions,
            attacker_label=attacker_label,
            initial_edge_set=aligned_initial_edge_set,
            target_edge_set=target_edge_set,
            title=title,
            is_target=False,
        )

    for axis in flat_axes[len(improvements):3]:
        axis.set_visible(False)

    draw_graph_panel(
        axis=flat_axes[3],
        graph=target_graph,
        positions=positions,
        attacker_label=attacker_label,
        initial_edge_set=initial_edge_set,
        target_edge_set=target_edge_set,
        title="Ground-truth graph",
        is_target=True,
    )

    legend_handles = [
        Line2D(
            [0],
            [0],
            color="#98A2B3",
            lw=2.2,
            label="Known edge",
        ),
        Line2D(
            [0],
            [0],
            color="#16875D",
            lw=3.0,
            label="Correctly added edge",
        ),
        Line2D(
            [0],
            [0],
            color="#C94747",
            lw=3.0,
            linestyle="--",
            label="Incorrectly added edge",
        ),
    ]

    figure.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        handlelength=2.8,
        columnspacing=2.0,
        handletextpad=0.7,
        bbox_to_anchor=(0.5, 0.025),
    )

    figure.subplots_adjust(
        left=0.035,
        right=0.965,
        top=0.89,
        bottom=0.13,
        hspace=0.29,
        wspace=0.08,
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    stem = output_directory / (
        f"seed_{seed_index:04d}"
        "_last_three_and_target"
    )

    figure.savefig(
        stem.with_suffix(".png"),
        dpi=400,
        bbox_inches="tight",
        facecolor="white",
    )

    figure.savefig(
        stem.with_suffix(".pdf"),
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(figure)

    print(
        f"Saved: {stem}.png and {stem}.pdf"
    )


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", default="cifar10", ) 
    parser.add_argument("--results-root", default="paper_results", ) 
    parser.add_argument("--reconstruction-dir", default=os.path.join( "paper_results", "DGD_reconstruction",),) 
    parser.add_argument("--output-dir", default=None, )

    parser.add_argument(
        "--run-index",
        "--seed-index",
        dest="run_index",
        type=int,
        default=None,
        help=(
            "Plot one seed only. "
            "By default, plot every seed."
        ),
    )

    parser.add_argument( "--layout-seed", type=int, default=42, )

    return parser.parse_args()


def main():
    args = parse_args()

    reconstruction_directory = (
        Path(args.reconstruction_dir)
        / args.dataset
    )

    (
        metadata,
        runs,
        reconstruction_directory,
    ) = load_experiment(
        dataset=args.dataset,
        results_root=args.results_root,
        reconstruction_dir=reconstruction_directory,
    )

    (
        target_graph,
        initial_graph,
        attacker_label,
    ) = build_graphs(metadata)

    output_directory = Path(
        args.output_dir
        or reconstruction_directory
        / "last_three_graph_plots"
    )

    selected_runs = runs

    if args.run_index is not None:
        selected_runs = [
            run
            for run in runs
            if int(run["run_index"])
            == args.run_index
        ]

        if not selected_runs:
            raise ValueError(
                f"Seed index {args.run_index} not found."
            )

    for run in selected_runs:
        plot_run(
            run=run,
            initial_graph=initial_graph,
            target_graph=target_graph,
            attacker_label=attacker_label,
            output_directory=output_directory,
            layout_seed=args.layout_seed,
        )


if __name__ == "__main__":
    main()
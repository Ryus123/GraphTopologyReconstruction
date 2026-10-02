#############################################################
#### Import
#############################################################
import argparse
import copy
import csv
import json
import os
import random
import time
from collections import deque
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

#############################################################
#### Classes and functions
#############################################################
class MNISTMLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Flatten(),

            nn.Linear(28 * 28, 2048),
            nn.ReLU(),

            nn.Linear(2048, 1024),
            nn.ReLU(),

            nn.Linear(1024, 1024),
            nn.ReLU(),

            nn.Linear(1024, 10),
        )

    def forward(self, x):
        return self.net(x)


def flat_params(model):
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def flat_grad(model):
    chunks = []
    for parameter in model.parameters():
        if parameter.grad is None:
            raise RuntimeError("One parameter does not have a gradient.")
        chunks.append(parameter.grad.detach().reshape(-1))
    return torch.cat(chunks)


def set_flat_params(model, vector):
    cursor = 0
    with torch.no_grad():
        for parameter in model.parameters():
            size = parameter.numel()
            parameter.copy_(vector[cursor:cursor + size].view_as(parameter))
            cursor += size


def gossip_matrix(graph, device):
    n = graph.number_of_nodes()
    degrees = dict(graph.degree())
    W = torch.zeros(n, n, dtype=torch.float32, device=device)
    for i, j in graph.edges():
        weight = 1.0 / (degrees[i] + degrees[j])
        W[i, j] = weight
        W[j, i] = weight
    for i in graph.nodes():
        W[i, i] = 1.0 - W[i].sum()
    if not torch.allclose(W, W.T, atol=1e-7):
        raise RuntimeError("W is not symmetric.")
    if not torch.allclose(W.sum(dim=1), torch.ones(n, device=device), atol=1e-6):
        raise RuntimeError("The matrix W is not row stochastic.")
    return W


def orthonormal_probes(q, dimension, seed, device):
    if q > dimension:
        raise ValueError("Number of probes exceeds model dimension.")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    gaussian = torch.randn(dimension, q, generator=generator, dtype=torch.float32)
    Q, _ = torch.linalg.qr(gaussian, mode="reduced")
    return Q.T.contiguous().to(device)


def build_dataset_and_model(experiment, data_root, download):
    if experiment == "mnist_mlp":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        train = datasets.MNIST(data_root, train=True, download=download, transform=transform)
        test = datasets.MNIST(data_root, train=False, download=download, transform=transform)
        return train, test, MNISTMLP()

    if experiment == "cifar10_resnet18":
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2470, 0.2435, 0.2616),
            ),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2470, 0.2435, 0.2616),
            ),
        ])
        train = datasets.CIFAR10(
            data_root, train=True, download=download, transform=train_transform
        )
        test = datasets.CIFAR10(
            data_root, train=False, download=download, transform=test_transform
        )
        model = models.resnet18(weights=None, num_classes=10)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        return train, test, model

    raise ValueError(f"Experience inconnue: {experiment}")


def make_node_loaders(dataset, n, batch_size, workers, seed, pin_memory):
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(len(dataset), generator=generator).numpy()
    shards = np.array_split(permutation, n)
    loaders = []
    for node, indices in enumerate(shards):
        subset = Subset(dataset, indices.tolist())
        loader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
            pin_memory=pin_memory,
            persistent_workers=workers > 0,
            drop_last=False,
            generator=torch.Generator().manual_seed(seed + node),
        )
        loaders.append(loader)
    return loaders


def next_batch(iterator, loader):
    try:
        batch = next(iterator)
    except StopIteration:
        iterator = iter(loader)
        batch = next(iterator)
    return iterator, batch


def evaluate_global(node_models, loader, criterion, device, max_batches):
    losses, accuracies = [], []
    for model in node_models:
        model.eval()
        total_loss = total_correct = total_count = 0
        with torch.inference_mode():
            for batch_index, (x, y) in enumerate(loader):
                if batch_index >= max_batches:
                    break
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                logits = model(x)
                total_loss += float(criterion(logits, y)) * y.numel()
                total_correct += int((logits.argmax(1) == y).sum())
                total_count += y.numel()
        model.train()
        losses.append(total_loss / max(total_count, 1))
        accuracies.append(total_correct / max(total_count, 1))
    return float(np.mean(losses)), float(np.mean(accuracies))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        choices=["mnist_mlp", "cifar10_resnet18"],
        default="mnist_mlp",
    )
    parser.add_argument("--nodes", type=int, default=9)
    parser.add_argument("--edge-probability", type=float, default=1 / 3)
    parser.add_argument("--malicious-node", type=int, default=0)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--projection-threshold", type=float, default=1e-4)
    parser.add_argument("--trigger-window", type=int, default=5)
    parser.add_argument("--min-rounds", type=int, default=100)
    parser.add_argument("--attack-deadline", type=int, default=1000)
    parser.add_argument("--max-rounds", type=int, default=5000)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--graph-seed", type=int, default=4332)
    parser.add_argument("--data", default=os.environ.get("DSDIR", "data"))
    parser.add_argument("--out", default=os.path.join("paper_results", "cifar10"))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--save-raw-observations", action="store_true")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )

    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument(
        "--lr-milestones",
        type=int,
        nargs="+",
        default=[40, 60],
    )
    parser.add_argument("--lr-gamma", type=float, default=0.1)

    args = parser.parse_args()

    if args.attack_deadline < args.min_rounds:
        raise ValueError("attack-deadline should be >= min-rounds.")
    if args.max_rounds < args.attack_deadline:
        raise ValueError("max-rounds should be >= attack-deadline.")

    defaults = {
        "mnist_mlp": {"lr": 1e-2, "batch_size": 128},
        "cifar10_resnet18": {"lr": 0.1, "batch_size": 128},
    }
    learning_rate = (
        args.lr if args.lr is not None else defaults[args.experiment]["lr"]
    )
    batch_size = args.batch_size or defaults[args.experiment]["batch_size"]

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA demande mais aucune GPU CUDA n'est visible.")

    n = args.nodes
    T_obs = 2 * n
    attacker = args.malicious_node
    output_directory = Path(args.out)
    output_directory.mkdir(parents=True, exist_ok=True)
    log_path = output_directory / "training.logs"

    def write_log(message):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    graph = nx.gnp_random_graph(n, args.edge_probability, seed=args.graph_seed)
    if not nx.is_connected(graph):
        raise RuntimeError("Le graphe genere n'est pas connexe.")
    neighbors = {i: sorted(graph.neighbors(i)) for i in graph.nodes()}
    attacker_neighbors = neighbors[attacker]
    W = gossip_matrix(graph, device)

    train_dataset, test_dataset, base_model = build_dataset_and_model(
        args.experiment, args.data, args.download
    )
    train_loaders = make_node_loaders(
        train_dataset,
        n,
        batch_size,
        args.workers,
        args.seed,
        device.type == "cuda",
    )
    train_iterators = [iter(loader) for loader in train_loaders]
    test_loader = DataLoader(
        test_dataset,
        batch_size=max(256, batch_size),
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )

    base_model = base_model.to(device)
    models_by_node = [copy.deepcopy(base_model).to(device) for _ in range(n)]
    criterion = nn.CrossEntropyLoss()
    dimension = flat_params(base_model).numel()

    momentum_buffers = torch.zeros(n, dimension, dtype=torch.float32, device=device, )
    steps_per_epoch = len(train_loaders[0])
    current_lr = learning_rate
    milestone_rounds = {
                            milestone * steps_per_epoch
                            for milestone in args.lr_milestones
                        }

    q = len(attacker_neighbors)
    probes = orthonormal_probes(q, dimension, args.seed + 1000, device)
    probe_of_neighbor = {u: k for k, u in enumerate(attacker_neighbors)}
    observed_nodes = [attacker] + attacker_neighbors

    trigger_history = deque(maxlen=args.trigger_window)
    attack_round = None
    trigger_reason = None
    projections = []
    raw_observations = []
    observation_rounds = []
    history = []


    write_log(
        f"experiment={args.experiment} device={device} n={n} D={dimension} "
        f"neighbors={attacker_neighbors} T_obs={T_obs} batch={batch_size}"
    )
    start_time = time.time()

    for t in range(args.max_rounds + T_obs + 1):
        X = torch.stack([flat_params(model) for model in models_by_node])
        gradients = []
        minibatch_losses = []

        for node, model in enumerate(models_by_node):
            train_iterators[node], (batch_x, batch_y) = next_batch(
                train_iterators[node], train_loaders[node]
            )
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            model.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            gradients.append(flat_grad(model))
            minibatch_losses.append(float(loss.detach()))
        gradients = torch.stack(gradients)

        # Y = X - learning_rate * gradients

        if args.experiment == "cifar10_resnet18":
            regularized_gradients = (gradients + args.weight_decay * X)
            momentum_buffers.mul_(args.momentum).add_(regularized_gradients)
            Y = X - current_lr * momentum_buffers

        else:
            Y = X - current_lr * gradients

        ordinary = torch.stack(
            [Y[u] for u in attacker_neighbors]
        )
        natural_abs = torch.abs(ordinary @ probes.T)
        noise_max = float(natural_abs.max())
        noise_mean = float(natural_abs.mean())

        if attack_round is None and t >= args.min_rounds:
            trigger_history.append(noise_max)
        window_max = (
            max(trigger_history)
            if len(trigger_history) == args.trigger_window
            else None
        )

        threshold_reached = (
            attack_round is None
            and window_max is not None
            and window_max <= args.projection_threshold
        )
        deadline_reached = attack_round is None and t >= args.attack_deadline
        if threshold_reached or deadline_reached:
            attack_round = t
            trigger_reason = (
                "projection_threshold" if threshold_reached else "forced_deadline"
            )
            write_log(
                f"ATTACK round={t} reason={trigger_reason} "
                f"window_max={window_max} threshold={args.projection_threshold:.1e}"
            )

        X_next = W @ Y

        # Launch attack
        if t == attack_round:
            model_param = torch.stack([ Y[node].detach().clone() for node in observed_nodes ])

            for receiver in attacker_neighbors:
                probe_index = probe_of_neighbor[receiver]
                X_next[receiver] -= (W[receiver, attacker] * Y[attacker])
                X_next[receiver] += probes[probe_index] #send amplified probes


        if attack_round is not None and 0 <= t - attack_round <= T_obs:
            O_t = torch.stack([
                X_next[u] for u in [attacker] + attacker_neighbors
                ]
            )
            Z_t = O_t @ probes.T
            projections.append(Z_t.detach().cpu().numpy().astype(np.float32))
            if args.save_raw_observations:
                raw_observations.append(
                    O_t.detach().cpu().numpy().astype(np.float32)
                )
            observation_rounds.append(t)

        for model, vector in zip(models_by_node, X_next):
            set_flat_params(model, vector)

        elapsed = time.time() - start_time
        record = {
            "round": t,
            "minibatch_loss_mean": float(np.mean(minibatch_losses)),
            "projection_noise_max": noise_max,
            "projection_noise_mean": noise_mean,
            "projection_window_max": window_max,
            "probe_injection": t == attack_round,
            "elapsed_seconds": elapsed,
            "test_loss": None,
            "test_accuracy": None,
        }

        should_evaluate = (
            t % args.eval_every == 0
            or t == attack_round
            or (attack_round is not None and t == attack_round + T_obs)
        )
        if should_evaluate:
            test_loss, test_accuracy = evaluate_global(
                models_by_node,
                test_loader,
                criterion,
                device,
                args.eval_batches,
            )
            record["test_loss"] = test_loss
            record["test_accuracy"] = test_accuracy

        history.append(record)

        if t % args.log_every == 0 or t == attack_round:
            write_log(
                f"round={t:05d} minibatch_loss={record['minibatch_loss_mean']:.4f} "
                f"test_loss={record['test_loss']} "
                f"test_accuracy={record['test_accuracy']} "
                f"noise_max={noise_max:.3e} window_max={window_max} "
                f"elapsed={elapsed / 60:.1f}min"
            )

        if attack_round is not None and t >= attack_round + T_obs:
            break

        if (args.experiment == "cifar10_resnet18" and (t + 1) in milestone_rounds):
            current_lr *= args.lr_gamma

            write_log(
                f"Learning rate updated to {current_lr:.2e} "
                f"at round {t + 1}"
            )

    if attack_round is None:
        raise RuntimeError("L'attaque n'a pas ete declenchee.")
    if len(projections) != T_obs + 1:
        raise RuntimeError(
            f"{len(projections)} observations, attendu {T_obs + 1}."
        )

    Z_A = np.stack(projections)
    save_payload = {
        "Z_A": Z_A,
        "probes": probes.detach().cpu().numpy().astype(np.float32),
        "projection_threshold": np.float32(args.projection_threshold),
        "neighbor_node_ids": np.asarray(attacker_neighbors, dtype=np.int64),
        "observation_steps": np.arange(T_obs + 1, dtype=np.int64),
        "absolute_rounds": np.asarray(observation_rounds, dtype=np.int64),
        "attack_round": np.int64(attack_round),
        "T_obs": np.int64(T_obs),
        "gossip_matrix": W.detach().cpu().numpy().astype(np.float32),
        "observed_node_ids": np.asarray(observed_nodes, dtype=np.int64, ),
        "model_parameters_at_attack": (model_param.detach().cpu().numpy().astype(np.float32)),
    }


    if args.save_raw_observations:
        save_payload["O_A"] = np.stack(raw_observations)
    np.savez_compressed(
        output_directory / "attacker_observations.npz", **save_payload
    )

    with (output_directory / "metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    metadata = {
        "experiment": args.experiment,
        "device": str(device),
        "nodes": n,
        "edge_probability": args.edge_probability,
        "graph_seed": args.graph_seed,
        "parameter_dimension": dimension,
        "T_obs": T_obs,
        "attack_round": attack_round,
        "trigger_reason": trigger_reason,
        "attack_deadline": args.attack_deadline,
        "projection_threshold": args.projection_threshold,
        "trigger_window": args.trigger_window,
        "attacker": attacker,
        "neighbors": attacker_neighbors,
        "Z_A_shape": list(Z_A.shape),
        "raw_observations_saved": args.save_raw_observations,
        "attack_message": "h_u",
    }
    (output_directory / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    rounds = [row["round"] for row in history]
    values = [row["projection_noise_max"] for row in history]
    figure, axis = plt.subplots(figsize=(9, 4))
    axis.semilogy(rounds, values, label="Maximum natural projection")
    axis.axhline(
        args.projection_threshold,
        color="black",
        linestyle=":",
        label="Threshold",
    )
    axis.axvline(
        attack_round,
        color="crimson",
        linestyle="--",
        label=f"Injection ({trigger_reason})",
    )
    axis.set_xlabel("Round")
    axis.set_ylabel("Absolute projection")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_directory / "projection_trigger.png", dpi=160)
    plt.close(figure)

    write_log(f"Done. Z_A={Z_A.shape}; sorties={output_directory.resolve()}")


if __name__ == "__main__":
    main()

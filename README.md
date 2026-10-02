# Graph Topology Reconstruction Attacks in Decentralized Learning

This repository contains the source code, experimental results, and plots associated with our graph topology reconstruction experiments.

## 1. Get the Code

This repository is anonymized for double-blind review. The code, experimental
results, and plots are available at:

https://anonymous.4open.science/r/GraphTopologyReconstruction-C133

To obtain a local copy, use the **Download Repository** button available on the
anonymous page (top-right menu), then extract the archive:

```bash
unzip GraphTopologyReconstruction-C133.zip
cd GraphTopologyReconstruction-C133
```

## 2. Installation

### Requirements

- Python 3.12.5 or later
- `pip`

Install the project dependencies with:

```bash
python -m pip install -r requirements.txt
```

## 3. Reproducing the Experiments

The repository already provides the results and plots. The following commands can be used to rerun the experiments from scratch.


### i. Reconstruction Benchmark

```bash
python -m graph_topology_attacks.algo.benchmark
```

### ii. Florentine Families Network with Two and Three Attackers

```bash
python -m graph_topology_attacks.algo.florentine_N_attackers
```

This experiment evaluates the reconstruction of the Florentine families network for the following attacker configurations:

- Two attackers: Medici and Guadagni
- Three attackers: Medici, Peruzzi, and Guadagni

### iii. Reconstruction with an Unknown Number of Edges

```bash
python -m graph_topology_attacks.algo.unknown_vs_known_edge_count_offline_prior \
    --nodes 9 \
    --time-budget 600 
```

### iv. Sensitivity of GNRPA with an Offline Prior to Graph Density

```bash
python -m graph_topology_attacks.algo.density_sensitivity_offline_prior
```

This experiment evaluates how the performance of GNRPA with an offline prior varies with the density of the target graph.

### v. Simulating Attacks on Decentralized Gradient Descent

```bash
python -u -m graph_topology_attacks.algo.gossip_DGD \
    --experiment cifar10_resnet18 \
    --device cuda \
    --data data \
    --out paper_results/cifar10 \
    --lr 3e-2 \
    --batch-size 128 \
    --workers 0 \
    --min-rounds 10 \
    --attack-deadline 47000 \
    --max-rounds 50000 \
    --projection-threshold 1e-4 \
    --log-every 1000 \
    --eval-every 1000
```

The CIFAR-10 dataset is downloaded automatically when this experiment is executed for the first time.

### vi. Graph Reconstruction from DGD Attack Observations

```bash
python -m graph_topology_attacks.algo.offline_prior_DGD_reconstruction \
    --dataset cifar10 \
    --results-root paper_results \
    --n-runs 50 \
    --time-budget 630 \
    --prior-simulations 300
```

This experiment reconstructs the communication topology from the observations produced by the decentralized gradient descent attack.

## 4. Repository Organization

### `utils.py`

Provides the shared utilities used throughout the project, including:

- attacker insertion and attacker-neighbor mapping;
- initialization of private probe vectors;
- construction of the gossip matrix;
- propagation of observations through gossip or DGD update rules;
- extraction of signals visible to attackers;
- loading and construction of graph instances;
- initialization and validation of partial reconstruction graphs;
- loading the learned models used by the offline prior;
- visualization of target and reconstructed graphs, rewards, and reconstruction errors.

### `metrics.py`

Contains the loss functions and graph-distance metrics used to compare a reconstructed graph with the target graph.

### `src/graph_topology_attacks/algo/nrpa_without_init.py`

Implements Nested Rollout Policy Adaptation (NRPA) for the graph topology reconstruction problem.

### `src/graph_topology_attacks/algo/offline_prior.py`

Implements GNRPA with an offline prior for the graph topology reconstruction problem. 

## 5. Data

The graph topologies used in the experiments are provided in:

```text
src/graph_topology_attacks/data/
```

No manual download is required for these graph instances.

The CIFAR-10 dataset is not stored in the repository. It is downloaded automatically when running the DGD attack simulation (command described in Section 3.v).

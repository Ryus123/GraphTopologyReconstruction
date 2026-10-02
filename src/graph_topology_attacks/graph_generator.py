#############################################################
#### Import
#############################################################
import networkx as nx
import numpy as np
import json
import os
from multiprocessing import Pool, cpu_count
from fedivertex import GraphLoader
import tqdm 

##############################################################
#### Functions
##############################################################
def graph_to_json(G: nx.Graph, n_nodes: int,) -> dict:
    def to_int(x):
        return int(x) if not isinstance(x, str) else int(x.split("_")[-1])
    
    edges = [(to_int(u), to_int(v)) for u, v in 
             sorted(G.edges(), key=lambda e: (to_int(e[0]), to_int(e[1])))]
    
    return {
        "n": n_nodes, "p": -1, "k": -1,
        "nodes_cap":  {str(i): -1 for i in range(n_nodes)},
        "nodes_cost": {str(i):  1 for i in range(n_nodes)},
        "edges_cost": [ {"i": u, "j": v, "cost": 1} for u, v in edges ],
        "edges": [ {"e": [(u), v], "weight": -1} for u, v in edges ],
        "m": G.number_of_edges()
    }


def generate_small_erdos_renyi_graph(n_nodes):
    dir_name = os.path.join("src", "graph_topology_attacks", "data", "ER")
    c = 1.6
    stats = {}
    stats[n_nodes] = {"number_of_graphs": 0, "edges_count": []}

    for i in range(100):
        graph_name = f"ER_{i}_w_{n_nodes}_nodes"
        c += 0.05 * (i//10)
        # p = c * (np.log(n_nodes) / n_nodes)
        p = 1/3
    
        ### --- Config target graph
        G = nx.gnp_random_graph(n_nodes, p)
        
        attempts = 0
        while not nx.is_connected(G):
            attempts += 1
            if attempts > 30:
                break
            
            G = nx.gnp_random_graph(n_nodes, p)

        else:
            stats[n_nodes]["number_of_graphs"] += 1
            stats[n_nodes]["edges_count"].append(G.number_of_edges())

            data = graph_to_json(G, n_nodes)
            path = os.path.join(dir_name, f"{graph_name}")
            with open(path, "w") as f:
                json.dump(data, f, indent=4)
    
    return stats


def generate_erdos_renyi_graph(args):
    n_nodes, m = args
    p = 2*m / (n_nodes * (n_nodes - 1))

    dir_name = os.path.join("src", "graph_topology_attacks", 
                            "data", f"medium_ER_{m}")
    stats = {}
    stats[f"{n_nodes}_{m}"] = {"number_of_graphs": 0, "edges_count": []}

    for i in range(100):
        graph_name = f"ER_{i}_w_{n_nodes}_nodes"
        G = nx.gnp_random_graph(n_nodes, p)
        
        attempts = 0
        while not nx.is_connected(G):
            attempts += 1
            if attempts > 30:
                break
            
            G = nx.gnp_random_graph(n_nodes, p)

        else:
            stats[f"{n_nodes}_{m}"]["number_of_graphs"] += 1
            stats[f"{n_nodes}_{m}"]["edges_count"].append(G.number_of_edges())

            data = graph_to_json(G, n_nodes)
            path = os.path.join(dir_name, f"{graph_name}")
            with open(path, "w") as f:
                json.dump(data, f, indent=4)
    
    return stats

def generate_barabasi_albert_graph(args):
    n_nodes, d = args

    dir_name = os.path.join("src", "graph_topology_attacks", "data", "Barabasi_Albert")
    stats = {}
    stats[f"{n_nodes}_{d}"] = {"number_of_graphs": 0, "edges_count": []}

    for i in range(100):
        graph_name = f"BA_{i}_w_{n_nodes}_nodes_d_{d}"    
        G = nx.barabasi_albert_graph(n_nodes, d)
        
        attempts = 0
        while not nx.is_connected(G):
            attempts += 1
            if attempts > 30:
                break
            
            G = nx.barabasi_albert_graph(n_nodes, d)

        else:
            stats[f"{n_nodes}_{d}"]["number_of_graphs"] += 1
            stats[f"{n_nodes}_{d}"]["edges_count"].append(G.number_of_edges())

            data = graph_to_json(G, n_nodes)
            path = os.path.join(dir_name, f"{graph_name}")
            with open(path, "w") as f:
                json.dump(data, f, indent=4)
    
    return stats

def create_fedivertex_subgraph(software:str, graph_type:str, n_min:int, n_max:int):
    dir_name = os.path.join("src", "graph_topology_attacks", "data", f"{software}_{graph_type}")
    loader = GraphLoader()
    G = loader.get_graph(software=software, graph_type=graph_type, index=-1, disable_tqdm=True)

    G = nx.convert_node_labels_to_integers(G, first_label=0, 
                                           ordering="default", 
                                           label_attribute="original_label" )

    node_names = nx.get_node_attributes(G, "original_label")

    stats_record = {}
    l_nodes = np.array(G.nodes())

    for centre in tqdm.tqdm(l_nodes, desc="Creating sub-graphs", unit="node"):
        cut = 0
        n_sub_nodes = 0
        last_cut = []
        while n_sub_nodes < n_min:
            cut += 1
            curr_cut = nx.single_source_shortest_path_length(G, centre, cutoff=cut).keys()
            if len(last_cut) == len(curr_cut):
                break 
            closed_nodes = list(curr_cut)
            H = G.subgraph(closed_nodes).copy()
            n_sub_nodes = H.number_of_nodes()
            last_cut = curr_cut

        if n_sub_nodes in list(range(n_min, n_max+1)):
            H_undirected = H.to_undirected()
            H_undirected = nx.convert_node_labels_to_integers(H_undirected, first_label=0, 
                                                              ordering="default", 
                                                              label_attribute="original_label" )

            n_nodes = H_undirected.number_of_nodes()

            res = stats_record.get(str(n_nodes), {"number_of_graphs": 0, "edges_count": []})
            res["number_of_graphs"] += 1
            res["edges_count"].append(H_undirected.number_of_edges())
            stats_record[str(n_nodes)] = res

            data = graph_to_json(H_undirected, n_nodes)
            path = os.path.join(dir_name, 
                                f"{software.upper()[0]}{software.upper()[0]}_{res["number_of_graphs"]}_w_{n_nodes}_nodes")
            with open(path, "w") as f:
                json.dump(data, f, indent=4)
    
    # Export stats
    stats_path = os.path.join(dir_name, f"stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats_record, f, indent=4)

    #Export mapping of node names
    node_names_path = os.path.join(dir_name, f"node_names.json")
    with open(node_names_path, "w") as f:
        json.dump(node_names, f, indent=4)

    return stats_record

##############################################################
#### Main
##############################################################
if __name__ == "__main__":
    small_nodes_size = [7, 8, 9]
    medium_nodes_size = [10, 11, 12]
    list_d = [2,3]
    list_m = [25]

    parameters = {
        "ER" : small_nodes_size.copy(),
        "medium_ER" : [(n, m) for n in medium_nodes_size for m in list_m],
        "small_BA" : [(n, 2) for n in small_nodes_size],
        "BA_large" : [(n, d) for n in medium_nodes_size for d in list_d],
        "Fedivertex" : [("bookwyrm", "federation", 8, 12), ("peertube", "follow", 8, 12)]
    }

    for type_graph, params in parameters.items():
        n_workers = min(len(params), max(1,cpu_count() - 4))
        with Pool(processes=n_workers) as pool:
            if type_graph == "ER":
                results = pool.map(generate_small_erdos_renyi_graph, params)
            elif type_graph == "medium_ER":
                results = pool.map(generate_erdos_renyi_graph, params)
            elif type_graph in ["small_BA", "BA_large"]:
                results = pool.map(generate_barabasi_albert_graph, params)
            elif type_graph == "Fedivertex":
                results = pool.starmap(create_fedivertex_subgraph, params)
    
        merged_stats = {}
        if type_graph != "Fedivertex":
            for stat in results:
                merged_stats.update(stat)

            stats_path = os.path.join("src", "graph_topology_attacks", "data", type_graph+"_stats.json")
            with open(stats_path, "w") as f:
                json.dump(merged_stats, f, indent=4)

    print("Benchmark data successfully generated and saved.")



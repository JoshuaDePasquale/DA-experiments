# Decomposability Algebra implementation
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List
import numpy as np
import networkx as nx


@dataclass
class DecomposabilityQuadruple:
    graph: nx.Graph
    state_function: Dict[str, float]
    rho: List[float]
    time: List[int]


def compute_path_distribution(graph: nx.DiGraph, max_depth: int = 3) -> Dict[str, float]:
    """Return probability distribution over DFS paths up to depth."""
    paths: List[str] = []
    for node in graph.nodes:
        for target in graph.nodes:
            if node == target:
                continue
            for path in nx.all_simple_paths(graph, source=node, target=target, cutoff=max_depth):
                paths.append("->".join(path))
    if not paths:
        return {}
    counts: Dict[str, int] = {}
    for p in paths:
        counts[p] = counts.get(p, 0) + 1
    total = sum(counts.values())
    return {p: c / total for p, c in counts.items()}


def entropy(dist: Dict[str, float]) -> float:
    probs = np.array(list(dist.values()), dtype=float)
    probs = probs[probs > 0]
    return -np.sum(probs * np.log(probs))


def kl_divergence(p: Dict[str, float], q: Dict[str, float]) -> float:
    keys = set(p) | set(q)
    p_arr = np.array([p.get(k, 0.0) for k in keys])
    q_arr = np.array([q.get(k, 0.0) for k in keys])
    # avoid zeros
    p_arr = np.where(p_arr == 0, 1e-12, p_arr)
    q_arr = np.where(q_arr == 0, 1e-12, q_arr)
    return float(np.sum(p_arr * np.log(p_arr / q_arr)))


def spt(p: Dict[str, float], q: Dict[str, float]) -> float:
    keys = set(p) | set(q)
    uniform = {k: 1.0 / len(keys) for k in keys}
    numerator = kl_divergence(p, uniform)
    denominator = kl_divergence(p, q)
    if denominator == 0:
        return 0.0
    return numerator / denominator


def classify_stt(rho_series: Iterable[float]) -> str:
    r = np.array(list(rho_series))
    if len(r) < 3:
        return "insufficient"
    first_deriv = np.gradient(r)
    second_deriv = np.gradient(first_deriv)
    if np.allclose(r, 0, atol=1e-3):
        return "collapse-resistant"
    if np.allclose(first_deriv, 0, atol=1e-3):
        return "flat"
    if np.all(second_deriv > 0) or np.all(second_deriv < 0):
        return "nonlinear"
    if np.allclose(second_deriv, 0, atol=1e-3):
        return "gradual"
    return "irregular"


# perturbation functions
def node_targeted(graph: nx.Graph, k: int = 1) -> nx.Graph:
    g = graph.copy()
    centrality = nx.betweenness_centrality(g)
    nodes = sorted(centrality, key=centrality.get, reverse=True)[:k]
    g.remove_nodes_from(nodes)
    return g


def node_random(graph: nx.Graph, k: int = 1) -> nx.Graph:
    g = graph.copy()
    nodes = list(g.nodes)
    np.random.shuffle(nodes)
    g.remove_nodes_from(nodes[:k])
    return g


if __name__ == "__main__":
    # demonstration with a small call graph
    G = nx.DiGraph()
    edges = [
        ("A", "B"),
        ("B", "C"),
        ("B", "D"),
        ("C", "E"),
        ("D", "E"),
        ("E", "F"),
    ]
    G.add_edges_from(edges)

    base_dist = compute_path_distribution(G)
    rho_series = [0.0]
    graphs = [G]
    for i in range(1, 5):
        perturbed = node_targeted(graphs[-1], k=1)
        graphs.append(perturbed)
        dist = compute_path_distribution(perturbed)
        rho = kl_divergence(base_dist, dist)
        rho_series.append(rho)

    print("rho:", rho_series)
    print("SPT:", spt(base_dist, compute_path_distribution(graphs[-1])))
    print("STT:", classify_stt(rho_series))

"""Minimal Decomposability Algebra examples.

This script implements simple utilities for computing the Decomposability
Algebra metrics without any third‑party dependencies.  It provides toy
demonstrations for each of the domains mentioned in the framework description.

The goal is not domain accuracy but rather to show how the framework can be
applied with lightweight data.  All graphs are tiny and synthetic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Set, Tuple
import math
import random


class Graph:
    """Very small directed graph implementation."""

    def __init__(self, directed: bool = True) -> None:
        self.directed = directed
        self.adj: Dict[str, Set[str]] = {}

    def add_edge(self, u: str, v: str) -> None:
        self.adj.setdefault(u, set()).add(v)
        self.adj.setdefault(v, set())
        if not self.directed:
            self.adj[v].add(u)

    def copy(self) -> "Graph":
        g = Graph(self.directed)
        g.adj = {n: set(nei) for n, nei in self.adj.items()}
        return g

    def nodes(self) -> List[str]:
        return list(self.adj.keys())

    def remove_node(self, n: str) -> None:
        self.adj.pop(n, None)
        for neighbours in self.adj.values():
            neighbours.discard(n)


def _dfs_paths(graph: Graph, src: str, tgt: str, cutoff: int) -> Iterable[List[str]]:
    path = [src]
    visited = {src}

    def _dfs(node: str) -> Iterable[List[str]]:
        if len(path) > cutoff:
            return
        if node == tgt and len(path) > 1:
            yield list(path)
        for neighbour in graph.adj.get(node, []):
            if neighbour in visited:
                continue
            visited.add(neighbour)
            path.append(neighbour)
            yield from _dfs(neighbour)
            path.pop()
            visited.remove(neighbour)

    yield from _dfs(src)


def compute_path_distribution(graph: Graph, max_depth: int = 3) -> Dict[str, float]:
    """Return probability distribution over simple paths."""

    paths: List[str] = []
    nodes = graph.nodes()
    for src in nodes:
        for tgt in nodes:
            if src == tgt:
                continue
            for p in _dfs_paths(graph, src, tgt, max_depth):
                paths.append("->".join(p))

    counts: Dict[str, int] = {}
    for p in paths:
        counts[p] = counts.get(p, 0) + 1

    total = sum(counts.values())
    if total == 0:
        return {}
    return {p: c / total for p, c in counts.items()}


def entropy(dist: Dict[str, float]) -> float:
    return -sum(p * math.log(p) for p in dist.values() if p > 0)


def kl_divergence(p: Dict[str, float], q: Dict[str, float]) -> float:
    keys = set(p) | set(q)
    total = 0.0
    for k in keys:
        pk = p.get(k, 0.0)
        qk = q.get(k, 1e-12)
        if pk == 0:
            continue
        if qk == 0:
            qk = 1e-12
        total += pk * math.log(pk / qk)
    return total


def spt(p: Dict[str, float], q: Dict[str, float]) -> float:
    keys = set(p) | set(q)
    if not keys:
        return 0.0
    uniform = {k: 1.0 / len(keys) for k in keys}
    num = kl_divergence(p, uniform)
    den = kl_divergence(p, q)
    return 0.0 if den == 0 else num / den


def classify_stt(series: Iterable[float]) -> str:
    r = list(series)
    if len(r) < 3:
        return "insufficient"
    first = [r[i + 1] - r[i] for i in range(len(r) - 1)]
    second = [first[i + 1] - first[i] for i in range(len(first) - 1)]

    if all(abs(v) < 1e-3 for v in r):
        return "collapse-resistant"
    if all(abs(v) < 1e-3 for v in first):
        return "flat"
    if all(v > 0 for v in second) or all(v < 0 for v in second):
        return "nonlinear"
    if all(abs(v) < 1e-3 for v in second):
        return "gradual"
    return "irregular"


def node_targeted(graph: Graph, k: int = 1) -> Graph:
    g = graph.copy()
    degrees = {n: len(nei) for n, nei in g.adj.items()}
    nodes = sorted(degrees, key=degrees.get, reverse=True)[:k]
    for n in nodes:
        g.remove_node(n)
    return g


def node_random(graph: Graph, k: int = 1) -> Graph:
    g = graph.copy()
    nodes = g.nodes()
    random.shuffle(nodes)
    for n in nodes[:k]:
        g.remove_node(n)
    return g


def demo_graph() -> Graph:
    g = Graph(True)
    edges = [
        ("A", "B"),
        ("B", "C"),
        ("B", "D"),
        ("C", "E"),
        ("D", "E"),
        ("E", "F"),
    ]
    for u, v in edges:
        g.add_edge(u, v)
    return g


def run_demo(name: str, graph: Graph, perturb) -> None:
    base = compute_path_distribution(graph)
    rho_series = [0.0]
    g = graph
    for _ in range(3):
        g = perturb(g)
        dist = compute_path_distribution(g)
        rho_series.append(kl_divergence(base, dist))

    final_dist = compute_path_distribution(g)
    print(f"{name} rho series: {rho_series}")
    print(f"{name} SPT: {spt(base, final_dist):.3f}")
    print(f"{name} STT: {classify_stt(rho_series)}")


def connectomic_example() -> None:
    g = Graph(False)
    edges = [("V1", "V2"), ("V2", "V3"), ("V2", "M1"), ("M1", "S1"), ("V3", "S1")]
    for u, v in edges:
        g.add_edge(u, v)
    run_demo("connectome", g, lambda gr: node_targeted(gr, k=1))


def gravitational_example() -> None:
    masses = {"Sun": 1000, "Earth": 10, "Moon": 1, "Jupiter": 100}
    g = Graph(True)
    g.add_edge("Sun", "Earth")
    g.add_edge("Earth", "Moon")
    g.add_edge("Sun", "Jupiter")

    base = {n: m / sum(masses.values()) for n, m in masses.items()}
    perturbed_masses = masses.copy()
    perturbed_masses["Jupiter"] = 80
    pert = {n: m / sum(perturbed_masses.values()) for n, m in perturbed_masses.items()}

    rho = kl_divergence(base, pert)
    print(f"gravitational rho: {[0.0, rho]}")
    print(f"gravitational SPT: {spt(base, pert):.3f}")
    print(f"gravitational STT: {classify_stt([0.0, rho, rho])}")


def software_example() -> None:
    run_demo("software", demo_graph(), lambda g: node_targeted(g, k=1))


def curricular_example() -> None:
    g = Graph(True)
    edges = [("Counting", "Addition"), ("Addition", "Multiplication"), ("Multiplication", "Algebra")]
    for u, v in edges:
        g.add_edge(u, v)
    run_demo("curriculum", g, lambda gr: node_random(gr, k=1))


def ecological_example() -> None:
    g = Graph(True)
    edges = [("Plants", "Herbivore"), ("Herbivore", "Carnivore"), ("Carnivore", "Apex")]
    for u, v in edges:
        g.add_edge(u, v)
    run_demo("ecology", g, lambda gr: node_random(gr, k=1))


def biological_example() -> None:
    g = Graph(True)
    edges = [("GeneA", "GeneB"), ("GeneB", "GeneC"), ("GeneA", "GeneC"), ("GeneC", "GeneD")]
    for u, v in edges:
        g.add_edge(u, v)
    run_demo("biology", g, lambda gr: node_targeted(gr, k=1))


def economic_example() -> None:
    g = Graph(True)
    edges = [("A", "B"), ("B", "C"), ("A", "C"), ("C", "D")]
    for u, v in edges:
        g.add_edge(u, v)
    run_demo("economy", g, lambda gr: node_random(gr, k=1))


if __name__ == "__main__":
    connectomic_example()
    gravitational_example()
    software_example()
    curricular_example()
    ecological_example()
    biological_example()
    economic_example()


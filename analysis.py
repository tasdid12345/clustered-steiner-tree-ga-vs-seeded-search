"""Compare a random-key GA (baseline) with a heuristic-seeded GA + local search (improved)
on simulated Clustered Steiner Tree instances generated with networkx. Seed 42 everywhere."""
import json, os, sys, time
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from networkx.algorithms.approximation import steiner_tree
from sklearn.model_selection import train_test_split

SEED = 42
SEEDS = [42, 43, 44]                                   # GA seeds for the test-set runs
N_GRAPHS, N_NODES = 30, 150
GENS, TOURN, LS_TRIALS = 40, 3, 3                      # same budget for both; LS only in improved
GRID = [(20, 0.02), (20, 0.1), (40, 0.02), (40, 0.1)]  # (population size, mutation rate)


def make_graph(seed):
    """Random clustered graph: 5-8 clusters, dense intra-cluster and sparse inter-cluster edges, one terminal per cluster."""
    rng = np.random.default_rng(seed)
    k = int(rng.integers(5, 9))
    sizes = rng.multinomial(N_NODES - 10 * k, [1 / k] * k) + 10   # every cluster has >= 10 nodes
    labels = np.repeat(np.arange(k), sizes)
    G = nx.Graph()
    G.add_nodes_from(range(N_NODES))
    for i in range(N_NODES):
        for j in range(i + 1, N_NODES):
            same = labels[i] == labels[j]
            if rng.random() < (0.15 if same else 0.01):
                G.add_edge(i, j, weight=int(rng.integers(1, 11) if same else rng.integers(5, 31)))
    for c in range(k):   # chain each cluster so it is connected
        nodes = np.where(labels == c)[0]
        for a, b in zip(nodes[:-1], nodes[1:]):
            G.add_edge(int(a), int(b), weight=int(rng.integers(1, 11)))
    for c in range(k - 1):   # chain clusters together so the whole graph is connected
        G.add_edge(int(np.where(labels == c)[0][0]), int(np.where(labels == c + 1)[0][0]),
                   weight=int(rng.integers(5, 31)))
    terminals = [int(rng.choice(np.where(labels == c)[0])) for c in range(k)]
    return G, terminals


def seed_trees(inst):
    """Heuristic trees as sets of edge indices: shortest-path tree, MST, and KMB Steiner approximation."""
    G, T, idx = inst["G"], inst["T"], inst["idx"]
    key = lambda a, b: idx[(min(a, b), max(a, b))]
    _, paths = nx.single_source_dijkstra(G, T[0], weight="weight")
    sp = set()
    for t in T:   # union of shortest paths from one terminal forms a subtree of an SPT
        p = paths[t]
        sp |= {key(a, b) for a, b in zip(p[:-1], p[1:])}
    mst = {key(a, b) for a, b in nx.minimum_spanning_tree(G, weight="weight").edges()}
    kmb = {key(a, b) for a, b in steiner_tree(G, T, weight="weight").edges()}
    return [sp, mst, kmb]


def prepare(G, T):
    """Flatten a graph into lists so every decode (for both models) uses the same data structure."""
    E = list(G.edges(data="weight"))
    inst = dict(G=G, T=T, n=G.number_of_nodes(), m=len(E),
                U=[u for u, _, _ in E], V=[v for _, v, _ in E], W=[w for _, _, w in E])
    inst["idx"] = {(min(u, v), max(u, v)): i for i, (u, v, _) in enumerate(E)}
    inst["seeds"] = seed_trees(inst)
    return inst


def decode(inst, keys):
    """Random-key decoder: Kruskal over edges sorted by key, then prune non-terminal leaves.
    Returns (tree cost, set of chosen edge indices)."""
    n, U, V, W, T = inst["n"], inst["U"], inst["V"], inst["W"], inst["T"]
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    chosen = []
    for e in np.argsort(keys, kind="stable"):
        ru, rv = find(U[e]), find(V[e])
        if ru != rv:
            parent[ru] = rv
            chosen.append(int(e))
            if len(chosen) == n - 1:
                break
    # prune: repeatedly drop leaves that are not terminals
    deg, adj = [0] * n, [[] for _ in range(n)]
    for e in chosen:
        for x in (U[e], V[e]):
            deg[x] += 1
            adj[x].append(e)
    is_term = [False] * n
    for t in T:
        is_term[t] = True
    alive = set(chosen)
    stack = [x for x in range(n) if deg[x] == 1 and not is_term[x]]
    while stack:
        x = stack.pop()
        for e in adj[x]:
            if e in alive:   # x is a leaf, so this is its only alive edge
                alive.discard(e)
                y = U[e] + V[e] - x
                deg[x] -= 1
                deg[y] -= 1
                if deg[y] == 1 and not is_term[y]:
                    stack.append(y)
                break
    return float(sum(W[e] for e in alive)), alive


def feasible(inst, S):
    """True if the selected edges connect all cluster terminals."""
    H = nx.Graph()
    H.add_nodes_from(inst["T"])
    H.add_edges_from((inst["U"][e], inst["V"][e]) for e in S)
    return all(nx.has_path(H, inst["T"][0], t) for t in inst["T"][1:])


def seed_keys(inst, S, rng):
    """Keys under which Kruskal picks the edges of tree S first (tree edges get keys in [0, 0.5))."""
    m = inst["m"]
    mask = np.zeros(m, bool)
    mask[list(S)] = True
    return np.where(mask, 0.5 * rng.random(m), 0.5 + 0.5 * rng.random(m))


def tournament(rng, cost):
    cand = rng.integers(0, len(cost), TOURN)
    return cand[np.argmin(cost[cand])]


def local_search(inst, keys, rng):
    """Key-space swap search: force a random non-tree edge in, or a random tree edge out; keep improvements."""
    cost, S = decode(inst, keys)
    for _ in range(LS_TRIALS):
        k2 = keys.copy()
        if rng.random() < 0.5:
            k2[rng.integers(inst["m"])] = -1.0      # force an edge into the tree
        else:
            if not S:
                continue
            k2[rng.choice(list(S))] = 2.0           # force a tree edge out
        c2, S2 = decode(inst, k2)
        if c2 < cost:
            keys, cost, S = k2, c2, S2
    return keys, cost


def run_ga(inst, pop_size, mut, improved, seed):
    """Random-key GA (tournament, one-point crossover, mutation, elitism).
    improved=True adds heuristic seeding of the initial population and local search on children."""
    rng = np.random.default_rng(seed)
    m = inst["m"]
    pop = rng.random((pop_size, m))
    if improved:   # heuristic trees occupy the first slots of the initial population
        for i, S in enumerate(inst["seeds"]):
            pop[i] = seed_keys(inst, S, rng)
    cost = np.array([decode(inst, k)[0] for k in pop])
    for _ in range(GENS):
        new_pop, new_cost = [pop[int(np.argmin(cost))].copy()], [cost.min()]
        while len(new_pop) < pop_size:
            p1, p2 = pop[tournament(rng, cost)], pop[tournament(rng, cost)]
            child = p1.copy()
            if rng.random() < 0.9:                  # one-point crossover on key vectors
                cut = int(rng.integers(1, m))
                child[cut:] = p2[cut:]
            mask = rng.random(m) < mut              # random-key mutation
            child[mask] = rng.random(int(mask.sum()))
            if improved:
                child, c = local_search(inst, child, rng)
            else:
                c = decode(inst, child)[0]
            new_pop.append(child)
            new_cost.append(c)
        pop, cost = np.array(new_pop), np.array(new_cost)
    b = int(np.argmin(cost))
    return float(cost[b]), decode(inst, pop[b])[1]


def main():
    graphs = [prepare(*make_graph(SEED + i)) for i in range(N_GRAPHS)]
    print(f"generated {len(graphs)} clustered graphs")
    train, test = train_test_split(np.arange(N_GRAPHS), test_size=1 / 3, random_state=SEED)
    print(f"split: {len(train)} train / {len(test)} test graphs")
    out = {}
    for name, improved in [("baseline", False), ("improved", True)]:
        # select population size and mutation rate by mean cost on TRAINING graphs only
        tune = {cfg: np.mean([run_ga(graphs[i], *cfg, improved, SEED)[0] for i in train]) for cfg in GRID}
        cfg = min(tune, key=tune.get)
        print(f"{name}: selected pop={cfg[0]}, mut={cfg[1]} (train mean cost {tune[cfg]:.2f})")
        costs, feas, times = np.zeros((len(SEEDS), len(test))), [], []
        for si, s in enumerate(SEEDS):   # test set used once per model, with 3 GA seeds
            for ti, i in enumerate(test):
                t0 = time.perf_counter()
                c, S = run_ga(graphs[i], *cfg, improved, s)
                times.append(time.perf_counter() - t0)
                costs[si, ti] = c
                feas.append(feasible(graphs[i], S))
        per_seed = costs.mean(axis=1)
        out[name] = dict(cfg=cfg, costs=costs, mean=float(per_seed.mean()), std=float(per_seed.std()),
                         feas=float(np.mean(feas)), time=float(np.mean(times)))
        print(f"{name}: test mean cost {out[name]['mean']:.2f} +- {out[name]['std']:.2f}, "
              f"feasible {out[name]['feas']:.2f}, {out[name]['time']:.2f}s/instance")

    b, im = out["baseline"], out["improved"]
    metrics = {
        "baseline_score": b["mean"], "improved_score": im["mean"],
        "improvement": b["mean"] - im["mean"],               # positive = improved has lower cost
        "baseline_score_std": b["std"], "improved_score_std": im["std"],
        "baseline_feasibility": b["feas"], "improved_feasibility": im["feas"],
        "baseline_time_per_instance_s": b["time"], "improved_time_per_instance_s": im["time"],
        "baseline_pop_size": b["cfg"][0], "improved_pop_size": im["cfg"][0],
        "baseline_mutation_rate": b["cfg"][1], "improved_mutation_rate": im["cfg"][1],
        "paired_win_rate_improved_le_baseline": float(np.mean(im["costs"] <= b["costs"])),
    }
    if not all(np.isfinite(v) for v in metrics.values()):
        sys.exit("ERROR: non-finite metrics")

    os.makedirs("figures", exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["baseline (random-key GA)", "improved (seeded GA + LS)"], [b["mean"], im["mean"]],
           yerr=[b["std"], im["std"]], capsize=6, color=["gray", "tab:blue"])
    ax.set_ylabel("mean test tree cost (lower is better)")
    fig.tight_layout()
    fig.savefig("figures/test_cost_bar.png", dpi=120)

    results = {
        "question": "Can a heuristic-seeded genetic algorithm with local search find lower-cost clustered "
                    "Steiner trees than a standard random-key genetic algorithm on the same simulated clustered networks?",
        "method": "Simulated clustered graphs (networkx, seeded); train/test split of graphs; GA population size and "
                  "mutation rate tuned on training graphs only; test cost averaged over GA seeds 42-44.",
        "dataset": {"name": "Simulated clustered networks (networkx random generator, seeds 42-71)",
                    "source_url": "https://networkx.org", "n_rows": N_GRAPHS, "n_train": len(train),
                    "n_test": len(test), "same_as_paper": False, "baseline_is_papers_method": False},
        "metric_name": "mean total tree cost on test graphs",
        "higher_is_better": False,
        "baseline": "Standard random-key GA: edge-priority keys, Kruskal-style decoding with terminal pruning, "
                    "tournament selection, one-point crossover, mutation.",
        "improved": "Same encoding and generation budget, but the initial population is seeded with shortest-path, "
                    "MST and KMB Steiner trees, and each child undergoes a key-space swap local search.",
        "metrics": metrics,
        "notes": "Simulated data, not the paper's dataset. Only 10 test graphs; std is over the 3 GA seeds "
                 "(ddof=0). Same generation count for both models, but local search adds extra decodes to the "
                 "improved model (compute is not identical). Feasibility is checked on each returned tree. "
                 "Tuning grid is small and uses a single seed on training graphs.",
    }
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("wrote results.json and figures/test_cost_bar.png")


if __name__ == "__main__":
    main()

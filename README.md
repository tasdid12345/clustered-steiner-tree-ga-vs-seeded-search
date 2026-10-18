# Heuristic-Seeded GA with Local Search for the Clustered Steiner Tree Problem

The Clustered Steiner Tree Problem asks for a low-cost tree that connects a required terminal in every cluster of a weighted network, possibly through intermediate nodes. The paper "Genetic algorithm based approach to solve the Clustered Steiner Tree Problem" (2024) proposes a genetic algorithm for this problem. Random-key GAs are a natural baseline for such encodings, but they can waste evaluations on poor initial trees, so this experiment checks whether heuristic seeding and local search help.

**Question:** Can a heuristic-seeded genetic algorithm with local search find lower-cost clustered Steiner trees than a standard random-key genetic algorithm on the same simulated clustered networks?

**Dataset:** Not the paper's dataset. 30 seeded random clustered networks (about 150 nodes, 5 to 8 clusters, weighted edges, one terminal per cluster) are generated with networkx. 20 graphs are used for training and tuning, and 10 are held out for testing.

**Baseline:** a standard random-key GA with edge-priority keys, Kruskal-style decoding with pruning to a tree covering all terminals, tournament selection, one-point crossover and mutation.

**Improved model:** the same encoding and generation budget, with the initial population seeded by shortest-path, MST and KMB Steiner trees, and a key-space swap local search applied to each child.

## How to run
```
pip install -r requirements.txt
python analysis.py
```

Results: results are written to `results.json` (with charts in `figures/`).

## Reference

This project is motivated by the following paper:

> TA Do, HB Ban, TTB Huynh, MT Le, BL Nguyen (2024). *Genetic algorithm based approach to solve the Clustered Steiner Tree Problem*. Evolutionary Intelligence. https://link.springer.com/article/10.1007/s12065-023-00848-w

Paper author: Prof. Ban Ha Bang, Hanoi University of Science and Technology (bangbh@soict.hust.edu.vn)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._

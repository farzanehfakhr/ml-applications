# Genetic Algorithm: TravOrder (Traveling Salesman Problem)

A genetic algorithm that solves the Traveling Salesman Problem using a custom chromosome representation designed specifically for this project ("TravOrder"). Built for MET CS 767 (Boston University), Assignment 5.

> Verified: running `travorder_ga.py` reproduces the exact routes and distances from the original notebook (18.4 for the 4-city example, 28.3 for the 6-city example) — this is a faithful transcription, not a re-derivation.

## The representation

Rather than encoding a route as a direct permutation of cities (which needs repair logic to stay valid after crossover/mutation), each chromosome is a list of **k-values**: gene *i* says "visit the *k*-th closest still-unvisited city at step *i*." Every chromosome of valid k-values automatically decodes to a valid route — no city is ever skipped or repeated, so crossover and mutation never produce an invalid tour.

```
reconstruct_route_from_chromosome([3, 1, 1], distance, start_index=0)
# at each step, picks the k-th nearest unvisited city -> decodes to a full route
```

## Algorithm

- **Fitness:** inverse of total route length (shorter route = higher fitness)
- **Selection:** k-way tournament selection
- **Crossover:** one-point crossover on the k-value chromosome
- **Mutation:** each gene independently has a chance to change to a different valid k-value
- **Elitism:** the best chromosome is always carried over to the next generation

## Results

| Example | Cities | Population | Generations | Best route | Distance |
|---|---|---|---|---|---|
| 1 | 4 (B, L, M, S) | 80 | 300 | B → S → M → L → B | **18.4** |
| 2 | 6 (A–F) | 120 | 400 | A → D → E → F → C → B → A | **28.3** |

Both runs converge well before the generation budget is used up (fitness plateaus by generation 50), suggesting the population size and mutation rate are generous for problems this small — the write-up's full discussion covers how these parameters would need to scale for larger city counts.

## Files

| File | Description |
|---|---|
| `travorder_ga.py` | Full GA implementation: distance matrix, chromosome encode/decode, fitness, selection, crossover, mutation, main loop, and both worked examples |

## Tools

Python, genetic algorithms, evolutionary computation

---
*Completed as part of graduate coursework (METCS767, Machine Learning) at Boston University.*

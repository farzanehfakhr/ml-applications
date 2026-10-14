"""
TravOrder: a genetic algorithm for the Traveling Salesman Problem.

"TravOrder" uses a custom chromosome representation: instead of encoding a
route as a permutation of city indices directly, each gene i is the k-th
closest *still-unvisited* city at step i (a "k-value" encoding). This keeps
every chromosome automatically valid (no repeated or missing cities), so
crossover and mutation never need repair logic.

Reconstructed from the author's own notebook screenshots (MET CS 767,
Assignment 5: Genetic Algorithms). Transcribed, not re-derived.
"""

import math
import random


# ----------- Data representation -----------
def build_distance_matrix(cities, edges):
    """
    cities: list of city names, e.g. ["B", "L", "M", "S"]
    edges:  dict mapping (city1, city2) -> distance (symmetric)
    returns: NxN matrix distance[i][j]
    """
    n = len(cities)
    index = {c: i for i, c in enumerate(cities)}
    distance = [[0.0] * n for _ in range(n)]

    for (c1, c2), d in edges.items():
        i, j = index[c1], index[c2]
        distance[i][j] = d
        distance[j][i] = d  # symmetric

    return distance


# ----------- Route reconstruction & length -----------
def reconstruct_route_from_chromosome(chromosome, distance, start_index=0):
    """
    chromosome: list of k-values, length N-1
                gene i = k-th closest unvisited city at step i
    distance:   NxN distance matrix
    start_index: index of starting city (default 0)
    returns: list of city indices in visit order (without final return)
    """
    n = len(distance)
    assert len(chromosome) == n - 1

    current = start_index
    unvisited = [i for i in range(n) if i != start_index]
    route = [current]

    for step, k in enumerate(chromosome):
        # At this step there should be n-1-step unvisited cities
        assert 1 <= k <= len(unvisited)

        # Sort unvisited cities by distance from current
        unvisited_sorted = sorted(unvisited, key=lambda j: distance[current][j])

        # Choose the k-th closest (1-based index)
        chosen = unvisited_sorted[k - 1]

        route.append(chosen)
        unvisited.remove(chosen)
        current = chosen

    return route


def route_length(route, distance):
    """
    route: list of city indices (visit order, no return)
    distance: NxN distance matrix
    returns: total tour length including return to start
    """
    total = 0.0
    n = len(route)
    for i in range(n - 1):
        total += distance[route[i]][route[i + 1]]
    total += distance[route[-1]][route[0]]  # return to start
    return total


# ----------- Fitness -----------
def fitness(chromosome, distance, start_index=0):
    route = reconstruct_route_from_chromosome(chromosome, distance, start_index)
    length = route_length(route, distance)
    return 1.0 / (length + 1e-9)  # avoid division by zero


# ----------- Chromosome initialization -----------
def random_chromosome(num_cities):
    """
    Create a random TravOrder chromosome for num_cities.
    Length = num_cities - 1
    Gene i is in [1, num_cities-1-i]
    """
    return [random.randint(1, num_cities - 1 - i)
            for i in range(num_cities - 1)]


def init_population(pop_size, num_cities):
    return [random_chromosome(num_cities) for _ in range(pop_size)]


# ----------- Selection (tournament) -----------
def tournament_selection(population, distance, k=3, start_index=0):
    """
    k-way tournament selection: pick k individuals at random
    and return the fittest.
    """
    competitors = random.sample(population, k)
    best = max(competitors, key=lambda c: fitness(c, distance, start_index))
    return best


# ----------- Crossover (TravOrder) -----------
def travorder_crossover(parent1, parent2):
    """
    One-point crossover on TravOrder chromosomes.
    Child = prefix from parent1 + suffix from parent2.
    """
    n = len(parent1)
    if n == 1:
        return parent1[:]  # trivial case

    # choose crossover point between 1 and n-1
    point = random.randint(1, n - 1)
    child = parent1[:point] + parent2[point:]
    return child


# ----------- Mutation (TravOrder) -----------
def travorder_mutation(chromosome, mutation_rate):
    """
    For each gene, with probability mutation_rate,
    change its k-value to a different valid k.
    """
    n = len(chromosome)        # chromosome length = N-1
    total_cities = n + 1       # N
    child = chromosome[:]

    for i in range(n):
        if random.random() < mutation_rate:
            # At step i, there are (N-1-i) unvisited cities
            k_max = total_cities - 1 - i

            # If there is only one choice, nothing to mutate
            if k_max == 1:
                continue

            current_k = child[i]
            new_k = current_k
            # pick a different integer in [1, k_max]
            while new_k == current_k:
                new_k = random.randint(1, k_max)

            child[i] = new_k

    return child


# ----------- GA main loop -----------
def best_individual(population, distance, start_index=0):
    best = max(population, key=lambda c: fitness(c, distance, start_index))
    best_fit = fitness(best, distance, start_index)
    return best, best_fit


def ga_travorder(distance,
                  start_index=0,
                  pop_size=80,
                  generations=300,
                  crossover_rate=0.9,
                  mutation_rate=0.03,
                  tournament_size=3,
                  elitism=True):
    """
    Run the TravOrder GA and return:
    - best chromosome
    - best route (as city indices)
    - best route length
    """
    num_cities = len(distance)
    population = init_population(pop_size, num_cities)

    best_chromosome, best_fit = best_individual(population, distance, start_index)

    for gen in range(generations):
        new_population = []

        # Elitism: keep the current best
        if elitism:
            new_population.append(best_chromosome[:])

        # Fill the rest of the new population
        while len(new_population) < pop_size:
            # Selection
            parent1 = tournament_selection(population, distance, tournament_size, start_index)
            parent2 = tournament_selection(population, distance, tournament_size, start_index)

            # Crossover
            if random.random() < crossover_rate:
                child = travorder_crossover(parent1, parent2)
            else:
                child = parent1[:]

            # Mutation
            child = travorder_mutation(child, mutation_rate)
            new_population.append(child)

        population = new_population

        # Update best solution
        candidate, candidate_fit = best_individual(population, distance, start_index)
        if candidate_fit > best_fit:
            best_chromosome, best_fit = candidate, candidate_fit

        # Optional: print progress every 50 generations
        if (gen + 1) % 50 == 0:
            print(f"Generation {gen+1}: best fitness = {best_fit:.6f}")

    # Final best route and length
    best_route = reconstruct_route_from_chromosome(best_chromosome, distance, start_index)
    best_len = route_length(best_route, distance)
    return best_chromosome, best_route, best_len


if __name__ == "__main__":
    random.seed(42)  # for reproducibility

    # ----- Example 1: 4-city dataset -----
    cities = ["B", "L", "M", "S"]
    edges = {
        ("B", "L"): 3.0,
        ("L", "M"): 4.5,
        ("M", "S"): 3.1,
        ("S", "L"): 5.7,
        ("B", "M"): 7.6,
        ("B", "S"): 7.8,
    }

    distance = build_distance_matrix(cities, edges)

    best_chrom, best_route, best_len = ga_travorder(
        distance,
        start_index=0,      # start at Boston (index 0)
        pop_size=80,
        generations=300,
        crossover_rate=0.9,
        mutation_rate=0.03,
        tournament_size=3,
        elitism=True
    )

    print("Best TravOrder chromosome:", best_chrom)
    print("Best route (by city index):", best_route)
    print("Best route (by city name):",
          " -> ".join(cities[i] for i in best_route) + f" -> {cities[best_route[0]]}")
    print("Total distance:", best_len)
    # Output: Best route (by city name): B -> S -> M -> L -> B, Total distance: 18.4

    # ----- Example 2: 6-city dataset (Section 6, generalization check) -----
    random.seed(42)
    cities6 = ["A", "B", "C", "D", "E", "F"]
    edges6 = {
        ("A", "B"): 4.2, ("A", "C"): 8.5, ("A", "D"): 5.7, ("A", "E"): 9.1, ("A", "F"): 6.3,
        ("B", "C"): 3.8, ("B", "D"): 7.4, ("B", "E"): 6.9, ("B", "F"): 8.2,
        ("C", "D"): 5.5, ("C", "E"): 7.2, ("C", "F"): 4.6,
        ("D", "E"): 3.9, ("D", "F"): 7.8,
        ("E", "F"): 6.1,
    }

    distance6 = build_distance_matrix(cities6, edges6)
    best_chrom6, best_route6, best_len6 = ga_travorder(
        distance6,
        start_index=0,
        pop_size=120,
        generations=400,
        crossover_rate=0.9,
        mutation_rate=0.03,
        tournament_size=3,
        elitism=True
    )

    print("\nBest TravOrder chromosome:", best_chrom6)
    print("Best route (by city index):", best_route6)
    print("Best route (by city name):",
          " -> ".join(cities6[i] for i in best_route6) + f" -> {cities6[best_route6[0]]}")
    print("Total distance:", best_len6)
    # Output: Best route (by city name): A -> D -> E -> F -> C -> B -> A, Total distance: 28.3

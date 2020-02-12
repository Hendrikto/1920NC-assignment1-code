import math

import matplotlib.pyplot as plt
import numpy as np
import numpy.random as random
from tqdm import tqdm

import tsp_instances


def sample_cuts(length):
    """
    Sample random different cuts.

    Args:
        length = [int] Length of interval to sample two cuts from

    Returns [int, int]:
        Two random different cuts in the interval [0, length].
    """
    cut1 = random.randint(length)
    cut2 = cut1 + 1 + random.randint(length - cut1)

    return cut1, cut2


def fill_missing_nodes(child, parent, cut1, cut2):
    """
    Add the missing nodes to the child from the parent after crossover.

    Args:
        child  = [ndarray] child candidate solution with missing nodes as -1
        parent = [ndarray] parent candidate solution which has the missing nodes
        cut1   = [int] first crossover cut
        cut2   = [int] second crossover cut

    Returns [ndarray]:
        Child candidate solution with the missing nodes from parent.
    """
    length = len(parent)
    child_idx = cut2 % length
    parent_idx = cut2 % length

    while child_idx >= cut2 or child_idx < cut1:
        node = parent[parent_idx]
        parent_idx = (parent_idx + 1) % length

        if node not in child:
            child[child_idx] = node
            child_idx = (child_idx + 1) % length

    return child


def crossover(parent1, parent2):
    """
    Cross over the orders of two parents to create two children.

    A random slice from parent1 is swapped with the same slice from parent2,
    whereafter the missing nodes are added to keep the order representation.

    Args:
        parent1 = [ndarray] first parent candidate solution
        parent2 = [ndarray] second parent candidate solution

    Returns [ndarray, ndarray]:
        Two children after crossover as Numpy arrays.
    """
    cut1, cut2 = sample_cuts(len(parent1))

    child1_cut = parent1[cut1:cut2]
    child2_cut = parent2[cut1:cut2]

    child1 = np.pad(child1_cut, (cut1, len(parent2) - cut2), mode='constant', constant_values=-1)
    child2 = np.pad(child2_cut, (cut1, len(parent1) - cut2), mode='constant', constant_values=-1)

    child1 = fill_missing_nodes(child1, parent2, cut1, cut2)
    child2 = fill_missing_nodes(child2, parent1, cut1, cut2)

    return child1, child2


def mutate(child):
    """
    Mutate a child candidate solution by swapping two nodes.

    Args:
        child = [ndarray] child candidate solution

    Returns [ndarray]:
        Mutated child candidate solution.
    """
    cut1, cut2 = sample_cuts(len(child) - 1)
    child[cut1], child[cut2] = child[cut2], child[cut1]

    return child


def fitness(route):
    """
    Compute fitness as total distance of route.

    Args:
        route = [ndarray] candidate solution

    Returns [float]:
        Fitness of route candidate solution.
    """
    total_distance = 0
    for node1_idx, node2_idx in zip(route, route[1:]):
        delta_x, delta_y = tsp[node1_idx] - tsp[node2_idx]
        total_distance += math.hypot(delta_x, delta_y)

    return total_distance


def tournament_selection(routes, num_parents):
    """
    Select parents for reproduction using binary tournament selection.

    Random pairs of routes are sampled and the route with a better fitness is
    taken as parent for reproduction.

    Args:
        routes      = [ndarray] candidate solutions
        num_parents = [int] number of parents to select

    Returns [ndarray]:
        Selected parents as a Numpy array.
    """
    routes_idx = random.randint(len(routes), size=2 * num_parents)
    route_pairs = zip(routes[routes_idx[::2]], routes[routes_idx[1::2]])

    parents = []
    for route1, route2 in route_pairs:
        if fitness(route1) < fitness(route2):
            parents.append(route1)
        else:
            parents.append(route2)

    return np.array(parents)


def create_children(parents, p_c, p_m, num_children):
    """
    Create children from parents using crossover and mutation.

    Random pairs of parents are sampled and crossover is applied with
    probability p_c. Otherwise, the children will be equal to the parents.
    Each resulting child is subsequently mutated with probability p_m.

    Args:
        parents      = [ndarray] parent candidate solutions
        p_c          = [float] probability of crossover
        p_m          = [float] probability of mutation
        num_children = [int] number of children to create

    Returns [ndarray]:
        Created children as a Numpy array.
    """
    parents_idx = random.randint(len(parents), size=num_children)
    parent_pairs = zip(parents[parents_idx[::2]], parents[parents_idx[1::2]])

    children = []
    for parent1, parent2 in parent_pairs:
        if random.rand() < p_c:
            children += crossover(parent1, parent2)
        else:
            children += [parent1.copy(), parent2.copy()]

    for i in range(num_children):
        if random.rand() < p_m:
            children[i] = mutate(children[i])

    return np.array(children)


def best_swap(route):
    """
    Perform local search that returns a potentially better route.

    Each subsequent pair of nodes is swapped in the given route. The route with
    the best fitness out of the given and swapped routes is returned.

    Args:
        route = [ndarray] candidate solution

    Returns [ndarray]:
        The given route, or a route with swapped subsequent nodes
        that has a better fitness.
    """
    best_route = route
    for i in range(len(route) - 1):
        swap_route = route.copy()

        swap_route[i], swap_route[i + 1] = swap_route[i + 1], swap_route[i]

        swap_fitness = fitness(swap_route)
        if swap_fitness < fitness(best_route):
            best_route = swap_route

    return best_route


def local_search(routes):
    """
    Perform local search on the given routes to potentially improve fitness.

    Args:
        routes = [ndarray] candidate solutions

    Returns [ndarray]:
        Potentially better routes found with local search.
    """
    better_routes = []
    for route in routes:
        better_route = best_swap(route)
        better_routes.append(better_route)

    return np.array(better_routes)


def algorithm(tsp, memetic, generations, population, p_c=1, p_m=0.001):
    """
    Run the genetic or memetic algorithm.

    Args:
        tsp         = [ndarray] Traveling Salesperson Problem instance
        memetic     = [bool] whether or not local search is used each iteration
        generations = [int] number of generations to run the algorithm
        population  = [int] number of candidate solutions in the population
        p_c         = [float] probability of crossover
        p_m         = [float] probability of mutation

    Returns:
        fitnesses  = [ndarray] array of fitnesses of all candidate solutions for
                               each generation
        best_route = [ndarray] best route that was found by the algorithm
    """
    # initialize the population of random candidate solutions
    routes = np.array([random.permutation(len(tsp)) for _ in range(population)])

    # run algorithm for a number of generations
    fitnesses = []
    best_route = routes[0]
    for _ in range(generations):
        if memetic:
            routes = local_search(routes)
        parents = tournament_selection(routes, num_parents=population)
        routes = create_children(parents, p_c, p_m, num_children=population)

        fitnesses.append([fitness(route) for route in routes])

        if min(fitnesses[-1]) < fitness(best_route):
            best_route = routes[np.argmin(fitnesses[-1])]

    return fitnesses, best_route


def run_algorithm(tsp, memetic, iterations, generations, population):
    """
    Run the genetic or memetic algorithm for a number of iterations.

    Args:
        tsp         = [ndarray] Traveling Salesperson Problem instance
        memetic     = [bool] whether or not local search is used each iteration
        iterations  = [int] number of times the algorithm is run
        generations = [int] number of generations to run the algorithm
        population  = [int] number of candidate solutions in the population

    Returns:
        fitnesses_runs = [ndarray] array of fitnesses of all candidate
                                   solutions for each generation for each run
        best_route     = [ndarray] best route that was found during all runs
    """
    fitnesses_runs = []
    best_route = random.permutation(len(tsp))

    for _ in tqdm(range(iterations)):
        fitnesses, route = algorithm(tsp, memetic, generations, population)

        fitnesses_runs.append(fitnesses)

        if fitness(route) < fitness(best_route):
            best_route = route

    return fitnesses_runs, best_route


def plot_results(results, generations, num_evaluations_per_generation):
    """
    Plot the results in one figure.

    The best, average, and worst fitness (averaged over runs) are displayed for
    each elapsed generation, as well as the best route that was found during all
    runs.

    Args:
        results [(fitnesses, best_route)] =
            fitnesses  = [ndarray] array of fitnesses of all candidate solutions
                                   for each generation for each run
            best_route = [ndarray] best route that was found during all runs
        generations                       = [int] number of generations the
                                                  algorithm was run
        num_evaluations_per_generation    = [int] number of fitness evaluations
                                                  during one generation
    """
    fitnesses, best_route = results

    fig, ax = plt.subplots(1, 2)

    ax[0].set_title('Worst, average, or best fitness for given evaluations')
    ax[0].set_xlabel('Evaluations')
    ax[0].set_ylabel('Worst, average, or best fitness')
    ax[0].plot(
        (np.arange(generations) + 1) * num_evaluations_per_generation,
        np.mean(np.max(fitnesses, axis=2), axis=0), label='Worst',
    )
    ax[0].plot(
        (np.arange(generations) + 1) * num_evaluations_per_generation,
        np.mean(fitnesses, axis=(0, 2)), label='Average',
    )
    ax[0].plot(
        (np.arange(generations) + 1) * num_evaluations_per_generation,
        np.mean(np.min(fitnesses, axis=2), axis=0), label='Best',
    )
    ax[0].legend()

    ax[1].scatter(tsp[:, 0], tsp[:, 1])
    ax[1].plot(tsp[best_route, 0], tsp[best_route, 1])
    ax[1].set_title('Best route')

    plt.show()


def exercise_six_a(tsp, num_runs=10, num_evaluations=100000, population=128):
    """
    Reproduce the results for Exercise 6(a).

    Plots the results on the TSP instance for the genetic and memetic
    algorithms, separately.

    Args:
        tsp             = [ndarray] Traveling Salesperson Problem instance
        num_runs        = [int] number of times the algorithms are run
        num_evaluations = [int] number of fitness evaluations per algorithm run
        population      = [int] number of candidate solutions in the population
    """
    for memetic in (False, True):
        if memetic:
            num_evaluations_per_generation = population * (2 + len(tsp))
            print('Running memetic algorithm...')
        else:
            num_evaluations_per_generation = population * 2
            print('Running genetic algorithm...')
        generations = num_evaluations // num_evaluations_per_generation + 1

        results = run_algorithm(tsp, memetic, num_runs, generations, population)

        plot_results(results, generations, num_evaluations_per_generation)


tsp = tsp_instances.bayg29
exercise_six_a(tsp)

tsp = tsp_instances.ulysses22
exercise_six_a(tsp)

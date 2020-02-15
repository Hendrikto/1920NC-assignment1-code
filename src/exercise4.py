import matplotlib.pyplot as plt
import numpy as np


def mutate(x, p):
    """
    Invert each of the bits in x with probability p.

    Args:
        x = [ndarray] candidate solution Boolean Numpy array
        p = [float] probability of mutation

    Returns [ndarray]:
        The mutated candidate solution.
    """
    invert = np.random.rand(len(x)) < p
    x_m = x ^ invert

    return x_m


def fitness(x):
    """
    Compute the fitness of the given candidate solution according to the
    Counting Ones problem.

    Args:
        x = [ndarray] candidate solution Boolean Numpy array

    Returns [int]:
        The fitness of the candidate solution.
    """
    return np.sum(x)


def genetic_algorithm(iterations=1500, length=100, replace=False):
    """
    Solve the Counting Ones problem with Simple (1+1)-GA for binary problems.

    Args:
        iterations = [int] number of iterations to run the algorithm
        length     = [int] length of the candidate solution bit strings
        replace    = [bool] whether or not x is replaced each iteration

    Returns [list]:
        List with fitness of each encountered candidate solution.
    """
    # probability of mutation
    p = 1 / length

    # initial candidate solution
    x = np.random.rand(length) < 0.5

    fitnesses = [fitness(x)]
    for _ in range(iterations):
        x_m = mutate(x, p)
        if fitness(x) < fitness(x_m) or replace:
            x = x_m

        fitnesses.append(fitness(x))

    return fitnesses


def exercise_four_a():
    """Reproduce results for exercise 4(a)."""
    # run algorithm
    fitnesses = genetic_algorithm()

    # plot results
    plt.plot(fitnesses)
    plt.title('Fitness of encountered candidate solutions')
    plt.xlabel('Iteration')
    plt.ylabel('Fitness')
    plt.show()
    plt.close()


def exercise_four_b():
    """Reproduce results for exercise 4(b)."""
    # set number of runs and length of candidate solutions
    num_runs = 10
    length = 100

    # for each run, determine whether the optimum was found
    optimum_found = 0
    for _ in range(num_runs):
        fitnesses = genetic_algorithm(length=length)
        best_fitness = max(fitnesses)
        optimum_found += best_fitness == length

    # print results
    print(f'Number of times the optimum was found: {optimum_found}')


def exercise_four_c():
    """Reproduce results for exercise 4(c)."""
    # set number of runs and length of candidate solutions
    num_runs = 10
    length = 100

    # for each run, determine whether the optimum was found
    optimum_found = 0
    for _ in range(num_runs):
        fitnesses = genetic_algorithm(length=length, replace=True)
        best_fitness = max(fitnesses)
        optimum_found += best_fitness == length

    # plot results
    plt.plot(fitnesses)
    plt.title('Fitness of encountered candidate solutions')
    plt.xlabel('Iteration')
    plt.ylabel('Fitness')
    plt.show()
    plt.close()

    # print results
    print(f'Number of times the optimum was found: {optimum_found}')


exercise_four_a()
exercise_four_b()
exercise_four_c()

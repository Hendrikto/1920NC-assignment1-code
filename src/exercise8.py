from math import (
    cos,
    exp,
    log,
    sin,
)
from operator import (
    add,
    mul,
    sub,
    truediv,
)

import matplotlib.pyplot as plt
import numpy as np
from deap import (
    algorithms,
    base,
    creator,
    gp,
    tools,
)

x_values = np.linspace(-1, 1, num=21)
y_values = np.array((
    0,
    -0.1629,
    -0.2624,
    -0.3129,
    -0.3264,
    -0.3125,
    -0.2784,
    -0.2289,
    -0.1664,
    -0.0909,
    0,
    0.1111,
    0.2496,
    0.4251,
    0.6496,
    0.9375,
    1.3056,
    1.7731,
    2.3616,
    3.0951,
    4,
))

primitive_set = gp.PrimitiveSet('exercise8', 1)
primitive_set.renameArguments(ARG0='x')

primitive_set.addPrimitive(add, 2)
primitive_set.addPrimitive(cos, 1)
primitive_set.addPrimitive(exp, 1)
primitive_set.addPrimitive(log, 1)
primitive_set.addPrimitive(mul, 2)
primitive_set.addPrimitive(sin, 1)
primitive_set.addPrimitive(sub, 2)
primitive_set.addPrimitive(truediv, 2)

creator.create('Fitness', base.Fitness, weights=(-1,))
creator.create('Individual', gp.PrimitiveTree, fitness=creator.Fitness)

toolbox = base.Toolbox()
toolbox.register('generate', gp.genGrow, primitive_set, min_=1, max_=5)
toolbox.register('individual', tools.initIterate, creator.Individual, toolbox.generate)
toolbox.register('population', tools.initRepeat, list, toolbox.individual)
toolbox.register('compile', gp.compile, pset=primitive_set)


def compute(individual, x_values):
    function_vec = np.vectorize(toolbox.compile(individual))
    return function_vec(x_values)


def evaluate(individual, x_values, y_values):
    """
    Evaluate an individual using sum of absolute errors.

    :returns: tuple of fitness scores
    """
    try:
        predictions = compute(individual, x_values)
    except (OverflowError, ValueError, ZeroDivisionError):
        return (float('inf'),)
    return (np.abs(y_values - predictions).sum(),)


# required functions for deap.algorithms.eaSimple
toolbox.register('evaluate', evaluate, x_values=x_values, y_values=y_values)
toolbox.register('mate', gp.cxOnePoint)
toolbox.register('select', tools.selTournament, tournsize=3)

statistics_fitness = tools.Statistics(lambda i: i.fitness.values[0])
statistics_fitness.register('minimum', min)
statistics_height = tools.Statistics(lambda i: (i.fitness.values[0], i.height))
statistics_height.register('best', lambda values: min(values)[1])
statistics_nr_nodes = tools.Statistics(lambda i: (i.fitness.values[0], len(i)))
statistics_nr_nodes.register('best', lambda values: min(values)[1])
statistics = tools.MultiStatistics(
    fitness=statistics_fitness,
    height=statistics_height,
    nr_nodes=statistics_nr_nodes,
)

population = toolbox.population(n=1000)
hall_of_fame = tools.HallOfFame(1)
population, log = algorithms.eaSimple(
    population,
    toolbox,
    cxpb=0.7,  # crossover probability
    mutpb=0,  # mutation probability
    ngen=50,
    halloffame=hall_of_fame,
    stats=statistics,
    verbose=True,
)

print('Fittest individual:', hall_of_fame[0])
print('Fitness:', toolbox.evaluate(hall_of_fame[0])[0])

plt.figure()
plt.title('Fittest Individual')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.scatter(x_values, y_values, marker='x', label='targets')
plt.plot(x_values, compute(hall_of_fame[0], x_values), label='fittest individual')
plt.legend()
plt.savefig('plots/exercise8-fittest_individual.pdf')
plt.show()
plt.close()

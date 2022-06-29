import random
import multiprocessing
from neural_net import NeuralNetwork, sigm, lin
from deap import base, creator, tools, algorithms
import numpy as np


nn = NeuralNetwork([1, 10, 10, 2], hidden_activ_func=sigm, output_activ_func=lin)
IND_SIZE = sum([(neuron.size+1) for layer in nn.layers for neuron in layer.neurons])


def evaluate(individual):
    nn.unpack(individual)
    out = nn.feed_forward([0.1])
    goal = [10.0, -10]
    o_g = zip(out, goal)
    return sum([abs(b-a) for a, b in o_g]),


def genetic_algo(pop_size=50, cross_over_pb=0.5, mutation_probability=0.2, n_generations=100, eval=evaluate):

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    toolbox = base.Toolbox()
    toolbox.register("attribute", random.random)
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                     toolbox.attribute, n=IND_SIZE)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", eval)

    pop = toolbox.population(n=pop_size)
    CXPB, MUTPB, NGEN = cross_over_pb, mutation_probability, n_generations

    hof = tools.HallOfFame(maxsize=10)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # Evaluate the entire population
    fitnesses = map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    for g in range(NGEN):
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # The population is entirely replaced by the offspring
        pop[:] = offspring
        hof.update(pop)
        print(stats.compile(pop))

    return pop, hof[0]


if __name__ == "__main__":
    random.seed(64)
    # Process Pool of 4 workers

    pop, best = genetic_algo(50, 0.5, 0.2, 300, evaluate)
    print(best)
    nn.unpack(best)
    print(nn.feed_forward([0.1]))

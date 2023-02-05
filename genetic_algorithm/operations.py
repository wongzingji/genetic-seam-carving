import numpy as np

def fitness(energy_map, dir, individual):
    rows, cols = energy_map.shape[:2]

    seam = create_seam(individual, dir)

    energy = 1.0  # distinguish from the 0 case -- out of bound
    if dir == 'row':
        for row, col in seam:
            # determine if it is out of bound
            if row < 0 or row >= rows:
                return 0.0
            energy += energy_map[row, col]
    elif dir == 'col':
        for row, col in seam:
            if col < 0 or col >= cols:
                return 0.0
            energy += energy_map[row, col]
    return energy ** np.e


# roulette - "stochastic acceptance"
# https://en.wikipedia.org/wiki/Fitness_proportionate_selection
def select(population, fitness):
    total = sum(fitness)

    selection_pool = []
    while len(selection_pool) < len(population):
        index = np.random.randint(low=0, high=len(population))
        fit = fitness[index]

        if fit > 0.0:  # ignore the out-of-bound seam
            probability = 1.0 - (fit / total)

            if random.random() < probability:  # smaller fitness value -- a larger prob to be selected
                selection_pool.append(population[index])

    return selection_pool


# single point
def cross(individual1, individual2):
    pi1, path1 = individual1
    pi2, path2 = individual2

    # keep track of pivot values
    pv1 = path1.pop(pi1)
    pv2 = path2.pop(pi2)

    point = np.random.randint(0, len(path1))

    path1[point:], path2[point:] = path2[point:], path1[point:]

    path1.insert(pi1, pv1)
    path2.insert(pi2, pv2)


# some kind of gaussian mutation
def mutate(individual, kernel):
    pivot, path = individual

    size = len(path)
    kernel_size = int(np.ceil(len(kernel) / 2))

    point = np.random.randint(low=0, high=size)
    window = [point + i for i in range(1 - kernel_size, kernel_size)]

    # print("mutate", kernel)

    for i in range(len(window)):
        index = window[i]
        if 0 <= index < size and index != pivot:
            if np.random.random() < kernel[i]:
                path[index] = np.random.randint(low=-1, high=2)


def gaussian(size, sigma):
    size = int(np.ceil(size / 2))
    r = range(1 - size, size)
    kernel = []

    for x in r:
        kernel.append(np.exp(-np.power(x, 2) / (2 * np.power(sigma, 2))))

    return kernel
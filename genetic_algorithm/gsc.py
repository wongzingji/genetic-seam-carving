import argparse
import functools
import multiprocessing
import random
from copy import deepcopy

import cv2
import numpy as np


def get_args():
    parser = argparse.ArgumentParser(description="Genetic Seam Carving")

    parser.add_argument("input", type=str, help="Input image")
    parser.add_argument("target_shape", type=int, nargs=2, help="Target shape in [row col] format")

    parser.add_argument("-show", action="store_true", help="Display visualization of seam carving process")

    return parser.parse_args()


# https://github.com/andrewdcampbell/seam-carving
def get_bool_mask(image_shape, seams):
    bool_mask = np.ones(shape=image_shape, dtype=np.bool)

    for seam in seams:
        for row, col in seam:
            # print(rows, cols, row, col, len(seam))
            bool_mask[row, col] = False

    return bool_mask


# https://github.com/andrewdcampbell/seam-carving
def visualize(image, bool_mask=None):
    display = image.astype(np.uint8)

    if bool_mask is not None:
        display[np.where(bool_mask == False)] = np.array([0, 0, 255])

    # display_resize = cv2.resize(display, (1000, 500))
    # cv2.imshow("visualization", display_resize)
    cv2.imshow("visualization", display)
    cv2.waitKey(100)

    return display

def visualize_seams(image, original_mask=None):
    display = image.astype(np.uint8)

    if original_mask is not None:
        display[np.where(original_mask < 0)] = np.array([0, 0, 255])
    return display

def remove_seams(image, bool_mask, dir):
    rows, cols = image.shape[:2]

    bool_mask = np.stack([bool_mask] * 3, axis=2)

    if dir == 'row':
        image = np.transpose(np.transpose(image)[np.transpose(bool_mask)].reshape(3, cols, rows - n))
    elif dir == 'col':
        image = image[bool_mask].reshape((rows, cols - n, 3))
    else:
        pass
        # alert

    return image


def create_seam(individual, dir):
    '''
    (pi, path) --> seam loc
    '''
    pivot, path = individual

    if dir == 'row':
        return [(f(pivot, path, i), i) for i in range(len(path))]
    elif dir == 'col':
        return [(i, f(pivot, path, i)) for i in range(len(path))]
    else:
        pass  # alert


def f(pivot, path, index):
    if index == pivot:
        return path[index]
    elif index > pivot:
        return path[index] + f(pivot, path, index - 1)
    elif index < pivot:
        return path[index] + f(pivot, path, index + 1)


def create_individual(image_shape, dir):
    '''
    individual -- pivot_loc, randomly generated path values
    dir: 0/1
    '''
    path = list(np.random.random_integers(low=-1, high=1, size=image_shape[1-dir]))
    pivot_index = np.random.randint(low=0, high=image_shape[1-dir]-1) # 原code没有-1
    pivot_value = np.random.randint(low=0, high=image_shape[dir])

    path[pivot_index] = pivot_value

    return pivot_index, path


def create_population(population_size, image_shape, dir):
    return [create_individual(image_shape, dir) for _ in range(population_size)]


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
def crossover(individual1, individual2):
    '''
    fixed pivot value (loc)
    '''
    pi1, path1 = individual1
    pi2, path2 = individual2

    # keep track of pivot values
    pv1 = path1.pop(pi1)
    pv2 = path2.pop(pi2)

    point = np.random.randint(0, len(path1))

    path1[point:], path2[point:] = path2[point:], path1[point:]

    path1.insert(pi1, pv1)
    path2.insert(pi2, pv2)

def spread(code, gap):
    '''
    code: ndarray
    gap: int (absolute value)
    return: ndarray
    '''
    indices = [np.random.randint(low=0, high=len(code)) for i in range(gap)]
    for idx in indices:
        code[idx] += -1

    code[np.argwhere(code<-1)] = -1

    return code

def ternary_encode(init_code, val):
    '''
    init_code: list of ones
    val: int
    '''
    code = spread(np.array(init_code), len(init_code)-val)
    while np.min(code) < -1:
        code = spread(code, np.sum(code)-val)

    code = list(code)
    return code


def crossover2(individual1, individual2):
    '''
    implemened by "An Improved Genetic Algorithms-based Seam Carving Method"
    flexible pivot value
    '''
    pi1, path1 = individual1
    pi2, path2 = individual2

    pv1 = path1.pop(pi1)
    pv2 = path2.pop(pi2)

    point1 = np.random.randint(0, len(path1))
    path1[point1:], path2[point1:] = path2[point1:], path1[point1:]

    if pv1 != pv2:
        # pivot ternary encoding
        code_l = [1]*max(pv1, pv2)  # larger
        code_s = deepcopy(code_l)  # smaller
        code_s = ternary_encode(code_s, min(pv1, pv2))

        point2 = np.random.randint(0, len(code_l))
        code_l[point2:], code_s[point2:] = code_s[point2:], code_l[point2:]

        if pv1 > pv2:
            path1.insert(pi1, np.sum(code_l))
            path2.insert(pi2, np.sum(code_s))
        else:
            path1.insert(pi1, np.sum(code_s))
            path2.insert(pi2, np.sum(code_l))
    else:
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

def new_generation(last_pop, energy_map, dir):
    # select mating pool
    fit_scores = pool.map(functools.partial(fitness, energy_map, dir), last_pop) # map each individual in the population
    selection_pool = select(last_pop, fit_scores)
    selection_pool = pool.map(deepcopy, selection_pool)

    # variations
    kernel = gaussian(21, 3.0)
    for individual1, individual2 in zip(selection_pool[::2], selection_pool[1::2]):  # sequential, 前后
        crossover2(individual1, individual2)
        mutate(individual1, kernel)
        mutate(individual2, kernel)

    return selection_pool


def process_seams(target_image, energy_func, dir):
    '''
    determine seams for each step in either direction;
    : param dir: dict item
    result: seam
    '''
    image_shape = target_image.shape[:2]
    # initialize
    population = create_population(pop_size, image_shape, dir[1])

    energy_map = energy_func(target_image)

    res = []
    for generation in range(num_generations):
        population = new_generation(population, energy_map, dir[0])
        fit_scores = pool.map(functools.partial(fitness, energy_map, dir[0]), population)
        res.append(min(fit_scores))

    # determine the solution
    # support multiple seams at a time
    fit_scores = pool.map(functools.partial(fitness, energy_map, dir[0]), population)
    elite = np.array(fit_scores).argsort()[:n]  # minimum n # 原来的代码用的max

    seams = []
    for idx in elite:
        seam = create_seam(population[idx], dir[0])
        seams.append(seam)

    return seams, res


def generate_original_mask(original_shape, dir):
    original_mask = np.zeros(original_shape)

    if dir == 'col':
        for i in range(original_shape[0]):
            original_mask[i, :] = range(original_shape[1])
    else:
        for i in range(original_shape[1]):
            original_mask[:, i] = range(original_shape[0])

    return original_mask.astype(int)


if __name__ == "__main__":
    args = get_args()

    # get image
    input_image = cv2.imread(args.input)
    input_image = input_image.astype(np.float64)
    target_image = input_image  # initialized target image
    original_shape = target_image.shape[:2]
    target_shape = tuple(args.target_shape)

    # create pool for multiprocessing
    pool = multiprocessing.Pool()

    # hyperparams
    # make hyperparameters an argument
    pop_size = 10
    num_generations = 20
    n = 1  # carve n seams at a time

    dirs = {'row':0, 'col':1}
    for item in dirs.items():
        original_mask = generate_original_mask(original_shape, item[0])
        while target_image.shape[item[1]] > target_shape[item[1]]:
            diff = target_image.shape[item[1]] - target_shape[item[1]]
            print("carving {} gap {}".format(item[0], diff))

            seams, res = process_seams(target_image, forward_energy, item)

            mask = get_bool_mask(target_image.shape[:2], seams)

            ##
            carved_mask = original_mask[original_mask>=0].reshape(target_image.shape[:2])
            r_indices = [item[0] for item in seams[0]]
            c_indices = [item[1] for item in seams[0]]
            vals = carved_mask[r_indices, c_indices]

            if item[0] == 'row':
                original_mask[vals,[j for j in range(len(vals))]] = -1
            else:
                original_mask[[i for i in range(len(vals))],vals] = -1

            if args.show:
                visualize(target_image, mask)

            target_image = remove_seams(target_image, mask, item[0])

        visual_seams = visualize_seams(input_image, original_mask)
        cv2.imwrite("visualize_seams.jpg", visual_seams)
    cv2.imwrite("target2.jpg", target_image)



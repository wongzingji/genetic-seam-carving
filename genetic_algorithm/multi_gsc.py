import argparse
import random
import numpy as np
import cv2
import multiprocessing
import functools
from copy import deepcopy
import math


def get_args():
    parser = argparse.ArgumentParser(description="Multi Genetic Seam Carving")

    parser.add_argument("input", type=str, help="Input image")
    parser.add_argument("target_shape", type=int, nargs=2, help="Target shape in [row col] format")

    parser.add_argument("-show", action="store_true", help="Display visualization of seam carving process")

    return parser.parse_args()

# https://github.com/andrewdcampbell/seam-carving
def get_bool_mask(image_shape, individual):
    bool_mask = np.ones(shape=image_shape, dtype=np.bool)

    for i in range(len(individual)):
        for j in individual[i]:
            bool_mask[i, j] = False

    return bool_mask

# https://github.com/andrewdcampbell/seam-carving
def visualize(image, bool_mask=None):
    display = image.astype(np.uint8)

    if bool_mask is not None:
        display[np.where(bool_mask == False)] = np.array([0, 0, 255])

    cv2.imshow("visualization", display)
    cv2.waitKey(100)

    return display


def remove(image, bool_mask):
    rows, cols = image.shape[:2]

    bool_mask = np.stack([bool_mask] * 3, axis=2)

    # if dir == 'row':
    #     image = np.transpose(np.transpose(image)[np.transpose(bool_mask)].reshape(3, cols, rows - n))
    # elif dir == 'col':
    image = image[bool_mask].reshape((rows, cols - n, 3))

    return image

def saliency_spectral_residual(img):
    _, saliency_map = cv2.saliency.StaticSaliencySpectralResidual_create().computeSaliency(img)
    return (saliency_map * 255).astype('uint8')

# energy map
def forward_energy(image):
    """
    Forward energy algorithm as described in "Improved Seam Carving for Video Retargeting"
    by Rubinstein, Shamir, Avidan.
    Vectorized code adapted from
    https://github.com/axu2/improved-seam-carving.
    """
    h, w = image.shape[:2]
    g_image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float64)

    energy = np.zeros((h, w))
    m = np.zeros((h, w))

    U = np.roll(g_image, 1, axis=0)
    L = np.roll(g_image, 1, axis=1)
    R = np.roll(g_image, -1, axis=1)

    cU = np.abs(R - L)
    cL = np.abs(U - L) + cU
    cR = np.abs(U - R) + cU

    for i in range(1, h):
        mU = m[i - 1]
        mL = np.roll(mU, 1)
        mR = np.roll(mU, -1)

        mULR = np.array([mU, mL, mR])
        cULR = np.array([cU[i], cL[i], cR[i]])
        mULR += cULR

        argmins = np.argmin(mULR, axis=0)
        m[i] = np.choose(argmins, mULR)
        energy[i] = np.choose(argmins, cULR)

    # vis = visualize(energy)
    # cv2.imwrite("forward_energy_demo.jpg", vis)

    return energy / 255.0

# individual
def create_individual(img_shape):
    '''
    return: a list of lists
    '''
    #TODO: n as pamarms
    loc_lst = []
    for i in range(img_shape[0]):
        loc_lst.append(list(np.random.choice(img_shape[1], 2, replace=False)))

    return loc_lst

# population
def create_population(pop_size, img_shape):
    return [create_individual(img_shape) for _ in range(pop_size)]


def n1_dist(val1, val2):
    dist = abs(val1 - val2)
    # 3种random walk的情况同样满足条件
    if dist <= 1:
        return 0
    return dist

def n2_dist(val1, val2):
    dist = pow(val1 - val2, 2)
    if abs(val1 - val2) <= 1:
        return 0
    return dist

def process_dist(indices1, indices2, dist_func):
    dist1 = dist_func(min(indices1), min(indices2)) + dist_func(max(indices1), max(indices2))
    dist2 = dist_func(min(indices1), max(indices2)) + dist_func(max(indices1), min(indices2))
    dist = min(dist1, dist2)
    return dist

def fitness(energy_map, dist_func, r, individual):
    '''
    energy + consecutive constraint
    '''
    # energy
    nrows = len(individual)  #######
    energy = 0
    reg = 0
    for i in range(nrows-1):
        col_idx1 = individual[i]
        col_idx2 = individual[i+1]
        # energy
        energy += sum([energy_map[i, col_idx1[0]], energy_map[i, col_idx1[1]]])
        # consecutive regularization -- penalize distance
        # absolute or quadratic
        reg += process_dist(col_idx1, col_idx2, dist_func)

    col_idx = individual[nrows-1]
    energy += sum([energy_map[nrows-1, col_idx[0]], energy_map[nrows-1, col_idx[1]]])  # last row

    return energy + r*math.log(reg)


def crossover(individual1, individual2):
    nrows = len(individual1)
    for i in range(nrows):
        indices1 = individual1[i]
        indices2 = individual2[i]
        if indices1 == indices2:
            pass
        else:
            indicator = random.choice([True, False])
            if indicator:
                individual1[i] = indices2
                individual2[i] = indices1


def mutation():
    pass

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


def new_generation(last_pop, energy_map, ratio):
    # select mating pool
    fit_scores = pool.map(functools.partial(fitness, energy_map, dist_func, r), last_pop)  # map each individual in the population
    # fit_func = functools.partial(fitness, energy_map)
    # fit_scores = [fit_func(ind) for ind in last_pop]

    selection_pool = select(last_pop, fit_scores)
    selection_pool = pool.map(deepcopy, selection_pool)

    # variations
    # kernel = gaussian(21, 3.0)
    for individual1, individual2 in zip(selection_pool[::2], selection_pool[1::2]):  # sequential, 前后
        crossover(individual1, individual2)
        # mutate(individual1, kernel)
        # mutate(individual2, kernel)

    # evaluation for new children
    fit_scores2 = pool.map(functools.partial(fitness, energy_map, dist_func, r), selection_pool)
    # fit_scores2 = [fit_func(ind) for ind in selection_pool]

    num = math.floor(ratio * len(last_pop))  # the number of parents kept
    # replace
    # at least one parent
    if num < 1:
        num = 1

    for idx in np.array(fit_scores2).argsort()[-num:]:
        selection_pool.pop(idx)
        fit_scores2.pop(idx)
    for idx in np.array(fit_scores).argsort()[:num]:
        selection_pool.append(last_pop[idx])
        fit_scores2.append(fit_scores[idx])

    return selection_pool, fit_scores2


if __name__ == '__main__':
    args = get_args()
    img = cv2.imread(args.input)
    img = img.astype(np.float64)  # initialized target image
    # img = cv2.cvtColor(cv2.imread(args.input), cv2.COLOR_BGR2RGB)
    target_shape = tuple(args.target_shape)

    pool = multiprocessing.Pool()

    nrows, ncols = img.shape[:2]

    energy_map = forward_energy(img)
    # energy_map = saliency_spectral_residual(img)
    r = 100  # penalize for consecuti# ve constraint
    pop_size = 5
    dist_func = n2_dist
    n = 2
    num_generation = 10

    while img.shape[1] > target_shape[1]:
        diff = img.shape[1] - target_shape[1]
        print("carving {} gap {}".format('col', diff))

        # initialize
        population = create_population(pop_size, img.shape)
        # evolution
        res = []
        for generation in range(num_generation):
            population, fit_scores = new_generation(population, energy_map, 0.3)  # functools.partial
            res.append(min(fit_scores))
        print(res)

        # determine the solution
        elite = population[np.array(fit_scores).argsort()[0]]

        bool_mask = get_bool_mask(img.shape[:2], elite)

        if args.show:
            visualize(img, bool_mask)

        img = remove(img, bool_mask)

    cv2.imwrite("target.jpg", img)
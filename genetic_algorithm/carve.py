import argparse
import functools
import multiprocessing
import random
from copy import deepcopy
from energy import *

import cv2
import numpy as np
from scipy import ndimage as ndi


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


# https://github.com/andrewdcampbell/seam-carving
def backward_energy(image):
    """
    Simple gradient magnitude energy map.
    """
    xgrad = ndi.convolve1d(image, np.array([1, 0, -1]), axis=1, mode='wrap')
    ygrad = ndi.convolve1d(image, np.array([1, 0, -1]), axis=0, mode='wrap')

    grad_mag = np.sqrt(np.sum(xgrad ** 2, axis=2) + np.sum(ygrad ** 2, axis=2))

    # vis = visualize(grad_mag)
    # cv2.imwrite("backward_energy_demo.jpg", vis)

    return grad_mag / 255.0




def create_seam(individual, dir):
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


def new_generation(last_pop, energy_map, dir):
    # select mating pool
    fit_scores = pool.map(functools.partial(fitness, energy_map, dir), last_pop) # map each individual in the population
    selection_pool = select(last_pop, fit_scores)
    selection_pool = pool.map(deepcopy, selection_pool)

    # TODO: figure this out
    # variations
    kernel = gaussian(21, 3.0)
    for individual1, individual2 in zip(selection_pool[::2], selection_pool[1::2]):  # sequential, 前后
        cross(individual1, individual2)  # return????
        mutate(individual1, kernel)
        mutate(individual2, kernel)

    return selection_pool  # selection_pool已经改变了??????


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

    for generation in range(num_generations):
        population = new_generation(population, energy_map, dir[0])

    # determine the solution
    # support multiple seams at a time
    fit_scores = pool.map(functools.partial(fitness, energy_map, dir[0]), population)
    elite = np.array(fit_scores).argsort()[:n]  # minimum n # 原来的代码用的max??????

    seams = []
    for idx in elite:
        seam = create_seam(population[idx], dir[0])
        seams.append(seam)

    return seams



if __name__ == "__main__":
    args = get_args()

    # get image
    input_image = cv2.imread(args.input)
    target_image = input_image.astype(np.float64) # initialized target image
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
        while target_image.shape[item[1]] > target_shape[item[1]]:
            diff = target_image.shape[item[1]] - target_shape[item[1]]
            print("carving {} gap {}".format(item[0], diff))

            seams = process_seams(target_image, forward_energy, item)
            mask = get_bool_mask(target_image.shape[:2], seams)

            if args.show:
                visualize(target_image, mask)

            target_image = remove_seams(target_image, mask, item[0])

    cv2.imwrite("target.jpg", target_image)


import os
import numpy as np
from sys import argv


def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def get_best_value(name):
    means = []
    stds = []
    names = []
    nums = []
    for file_ in os.listdir(name):
        results = []
        num = 0
        try:
            with open(name + '/' + file_, 'r') as file:
                for line in file:
                    line = line.split(' ')
                    if (isfloat(line[0])):
                        num += 1
                        results.append(float(line[0]))
            nums.append(num)
            results = np.array(results)
            means.append(np.mean(results))
            stds.append(np.std(results))
            names.append(file_)
        except:
            means.append(-1000.)
            names.append(file_)
            print('Couldn\'t open file with name: ', file_)

    means = np.array(means)
    stds = np.array(stds)
    means_sorted = np.sort(means)
    args = np.argsort(means)

    for i in range(1, 1 + len(means)):
        print(means_sorted[-i], stds[args[-i]], nums[args[-i]], names[args[-i]])


name = argv[1]
get_best_value(name)

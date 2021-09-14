import os
import math


def prepare_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def nice_print_dict(dictionary):
    print("\t".join("{} {}".format(k, v) for k, v in dictionary.items()))


def list_to_chunks(lst, n):
    res = []
    idx = range(0, len(lst)+1, math.ceil(len(lst)/n))
    for i, j in zip(idx[:-1], idx[1:]):
        res.append(lst[i:j])
    return res

def nice_print_dict(dictionary):
    print("\t".join("{} {}".format(k, v) for k, v in dictionary.items()))

import math
def list_to_chunks (lst, n):
    res = []
    idx = range(0, len(lst)+1, math.ceil(len(lst)/n))
    for i,j in zip(idx[:-1], idx[1:]):
        res.append(lst[i:j])
    return res

import os
import numpy as np

def generate():
    return np.random.randint(100_000, 999_999)

print(generate())

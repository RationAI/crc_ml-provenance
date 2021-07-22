from concurrent.futures import ProcessPoolExecutor
from random import randint
NUMPARALLELPROCS=10

def doWork(arg, a) -> int :
    return randint(0, 1000)

with ProcessPoolExecutor() as pool:
    futures = []
    results = []

    a = [1,2,3,4]

    for i in range(1,1000):
        futures.append(pool.submit(doWork, a, "b"))
    for f in futures:
        results.append(f.result())

    print(results)


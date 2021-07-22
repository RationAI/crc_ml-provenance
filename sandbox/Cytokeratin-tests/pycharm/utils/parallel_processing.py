from concurrent.futures import ProcessPoolExecutor


def _process_args_parallel(pool, func, args_list, **kvargs):
    futures = []
    results = []

    if pool==None: pool = ProcessPoolExecutor()

    for block in args_list:
        if len(kvargs.keys()) > 0:
            futures.append(pool.submit(func, *block, **kvargs))
        else: futures.append(pool.submit(func, *block))

    for f in futures:
        results.append(f.result())

    print("All processes finished!")
    return results
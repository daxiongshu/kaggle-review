import time
import numpy as np
import os
import random
import multiprocessing
import sys

def parallel_run(func,data):
    p = multiprocessing.Pool()
    results = p.imap(func, data)
    num_tasks = len(data)
    while (True):
        completed = results._index
        print("\r--- Completed {:,} out of {:,}".format(completed, num_tasks),end="")
        sys.stdout.flush()
        time.sleep(1)
        if (completed == num_tasks):
            break
    p.close()
    p.join()
    return results

def parallel_run1(func,data):
    # sequential sanity check
    for d in data:
        func(d)


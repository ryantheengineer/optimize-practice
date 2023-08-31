# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 12:39:28 2023

@author: Ryan.Larson
"""

import multiprocessing
import time

# Dummy function to simulate a time-consuming task
def time_consuming_task(task_id, arg1, arg2):
    print(f"Task {task_id} started with arguments {arg1} and {arg2}", flush=True)
    time.sleep(2)  # Simulate task duration
    print(f"Task {task_id} completed", flush=True)
    return task_id

def main():
    num_tasks = 20
    
    # Without multiprocessing
    start_time = time.time()
    results = [time_consuming_task(i, f"Argument-{i}", f"Value-{i}") for i in range(num_tasks)]
    end_time = time.time()
    print(f"Without multiprocessing: {end_time - start_time:.2f} seconds")

    # With multiprocessing using starmap
    start_time = time.time()
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    arg_tuples = [(i, f"Argument-{i}", f"Value-{i}") for i in range(num_tasks)]
    results_mp = pool.starmap(time_consuming_task, arg_tuples)
    pool.close()
    pool.join()
    end_time = time.time()
    print(f"With multiprocessing: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()



# import multiprocessing
# import time

# # Dummy function to simulate a time-consuming task
# def time_consuming_task(task_id):
#     print(f"Task {task_id} started", flush=True)
#     time.sleep(2)  # Simulate task duration
#     print(f"Task {task_id} completed", flush=True)
#     return task_id

# def main():
#     num_tasks = 20

#     # Without multiprocessing
#     start_time = time.time()
#     results = [time_consuming_task(i) for i in range(num_tasks)]
#     end_time = time.time()
#     print(f"Without multiprocessing: {end_time - start_time:.2f} seconds")

#     # With multiprocessing
#     start_time = time.time()
#     pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
#     results_mp = pool.map(time_consuming_task, range(num_tasks))
#     pool.close()
#     pool.join()
#     end_time = time.time()
#     print(f"With multiprocessing: {end_time - start_time:.2f} seconds")

# if __name__ == "__main__":
#     main()

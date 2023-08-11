# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 14:38:51 2023

@author: Ryan.Larson
"""

import random
import multiprocessing
import time
import queue
from NSGA_II_Example import main_optimization

def throw_pokeball(params):
    HPmax = params[0]
    HPcurrent = params[1]
    rate = params[2]
    bonusball = params[3]
    bonusstatus = params[4]
    
    a = ((3*HPmax - 2*HPcurrent) * rate * bonusball * bonusstatus) / (3*HPmax)
    p = (65536 / (255 / a)**(1/4))/65536
    
    nchecks = random.randint(1,3)
    catch = True
    for i in range(nchecks):
        rnd = random.uniform(0,1)
        if rnd > p:
            catch = False
            break
    
    return catch

def worker_function(result_queue):
    while True:
        fitnesses = main_optimization()
        result_queue.put(fitnesses)

# def worker_function(param_queue, result_queue):
#     while True:
#         try:
#             params = param_queue.get_nowait()
#         except queue.Empty:
#             break
#         catch = throw_pokeball(params)
#         result_queue.put((params, catch))



def random_params():
    HPmax = random.randint(50,300)
    while True:
        HPcurrent = random.randint(10,300)
        if HPcurrent <= HPmax:
            break
    rate = random.randint(1,255)
    bonusball = random.choice([1, 1.5, 3, 3.5, 4, 5, 8])
    bonusstatus = random.choice([1, 2.5, 1.5])
    params = [HPmax, HPcurrent, rate, bonusball, bonusstatus]
    return params

def multiprocessing_pokemon_main(population):
    start_time = time.time()
    num_processes = multiprocessing.cpu_count()  # Number of available CPU cores
    param_queue = multiprocessing.Queue()
    result_queue = multiprocessing.Queue()
    
    for design in population:
        param_queue.put(design)
    
    # Create and start worker processes
    processes = []
    for _ in range(num_processes):
        process = multiprocessing.Process(target=worker_function, args=(param_queue, result_queue))
        process.start()
        processes.append(process)

    # Wait for all worker processes to finish
    for process in processes:
        process.join()

    # Retrieve results from the result queue
    results = []
    while not result_queue.empty():
        design, fitness = result_queue.get()
        results.append((design, fitness))
    
    print(f"--- {time.time() - start_time} seconds for multiprocessing---")
    return results

def multiprocessing_NSGA_II_main(ndesigns):
    start_time = time.time()
    num_processes = multiprocessing.cpu_count()  # Number of available CPU cores
    # param_queue = multiprocessing.Queue()
    result_queue = multiprocessing.Queue()
    
    # for design in range(ndesigns):
    #     param_queue.put(design)
    
    # Create and start worker processes
    processes = []
    for _ in range(num_processes):
        process = multiprocessing.Process(target=main_optimization)
        process.start()
        processes.append(process)

    # Wait for all worker processes to finish
    for process in processes:
        process.join()

    # # Retrieve results from the result queue
    # results = []
    # while not result_queue.empty():
    #     fitness = result_queue.get()
    #     results.append((fitness))
    
    print(f"--- {time.time() - start_time} seconds for multiprocessing---")
    # return results
    
def straight_main(ndesigns):
    start_time = time.time()
    results = [main_optimization() for _ in range(ndesigns)]
    print(f"--- {time.time() - start_time} seconds for straight calculation ---")
    # return results
    
    
if __name__ == "__main__":
    ncases = 20
    # print("Generating population...")
    # population = [random_params() for _ in range(ncases)]
    
    print("\nStarting multiprocessing...")
    multiprocessing_NSGA_II_main(ncases)
    
    print("\nStarting straight calculation...")
    straight_main(ncases)
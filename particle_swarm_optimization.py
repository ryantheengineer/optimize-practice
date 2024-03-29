# -*- coding: utf-8 -*-
"""
Created on Tue May 24 15:09:17 2022

@author: Ryan.Larson
"""

# From https://www.youtube.com/watch?v=8xycqWWqz50

import random
import math
import matplotlib.pyplot as plt
import numpy as np
# ------------------------------------------------------------------------------
# TO CUSTOMIZE THIS PSO CODE TO SOLVE UNCONSTRAINED OPTIMIZATION PROBLEMS, CHANGE THE PARAMETERS IN THIS SECTION ONLY:
# THE FOLLOWING PARAMETERS MUST BE CHANGED.
def objective_function(X):
    """
    Example objective function

    Parameters
    ----------
    X : TYPE
        DESCRIPTION.

    Returns
    -------
    y : TYPE
        DESCRIPTION.

    """
    A = 10
    y = A*2 + sum([(x**2 - A * np.cos(2 * math.pi * x)) for x in X])
    return y
  
# ------------------------------------------------------------------------------
class Particle:
    def __init__(self, bounds, initial_fitness):
        self.particle_position = []  # particle position
        self.particle_velocity = []  # particle velocity
        self.local_best_particle_position = []  # best position of the particle
        self.fitness_local_best_particle_position = initial_fitness  # initial objective function value of the best particle position
        self.fitness_particle_position = initial_fitness  # objective function value of the particle position
  
        for i in range(nv):
            self.particle_position.append(
                random.uniform(bounds[i][0], bounds[i][1]))  # generate random initial position
            self.particle_velocity.append(random.uniform(-1, 1))  # generate random initial velocity
  
    def evaluate(self, objective_function):
        self.fitness_particle_position = objective_function(self.particle_position)
        if mm == -1:
            if self.fitness_particle_position < self.fitness_local_best_particle_position:
                self.local_best_particle_position = self.particle_position # update the local best
                self.fitness_local_best_particle_position = self.fitness_particle_position  # update the fitness of the local best
        if mm == 1:
            if self.fitness_particle_position > self.fitness_local_best_particle_position:
                self.local_best_particle_position = self.particle_position
                self.fitness_local_best_particle_position = self.fitness_particle_position
                
    def update_velocity(self, global_best_particle_position, w, c1, c2):
        for i in range(nv):
            r1 = random.random()
            r2 = random.random()
            
            cognitive_velocity = c1 * r1 * (self.local_best_particle_position[i] - self.particle_position[i])
            social_velocity = c2 * r2 * (global_best_particle_position[i] - self.particle_position[i])
            self.particle_velocity[i] = w * self.particle_velocity[i] + cognitive_velocity + social_velocity
            
    def update_position(self, bounds):
        for i in range(nv):
            self.particle_position[i] = self.particle_position[i] + self.particle_velocity[i]
            
            # check and repair to satisfy the upper bounds
            if self.particle_position[i] > bounds[i][1]:
                self.particle_position[i] = bounds[i][1]
            # check and repair to satisfy the lower bounds
            if self.particle_position[i] < bounds[i][0]:
                self.particle_position[i] = bounds[i][0]


def optimize_particle_swarm(mm, objective_function, bounds, particle_size, iterations, w, c1, c2):              
    # -----------------------------------------------------------------------------
    if mm == -1:
        initial_fitness = float("inf") # for minimization problem
    if mm == 1:
        initial_fitness = -float("inf") # for maximization problem
        
    # -----------------------------------------------------------------------------
    # Visualization
    fig = plt.figure()
    ax = fig.add_subplot()
    fig.show()
    # Begin solving the optimization problem here
    fitness_global_best_particle_position = initial_fitness
    global_best_particle_position = []
    # Create the group of swarm particles
    swarm_particle = []
    for i in range(particle_size):
        swarm_particle.append(Particle(bounds, initial_fitness))
    A = []
    
    for i in range(iterations):
        for j in range(particle_size):
            swarm_particle[j].evaluate(objective_function)
            
            if mm == -1:
                if swarm_particle[j].fitness_particle_position < fitness_global_best_particle_position:
                    global_best_particle_position = list(swarm_particle[j].particle_position)
                    fitness_global_best_particle_position = float(swarm_particle[j].fitness_particle_position)
            if mm == 1:
                if swarm_particle[j].fitness_particle_position > fitness_global_best_particle_position:
                    global_best_particle_position = list(swarm_particle[j].particle_position)
                    fitness_global_best_particle_position = float(swarm_particle[j].fitness_particle_position)
        for j in range(particle_size):
            swarm_particle[j].update_velocity(global_best_particle_position, w, c1, c2)
            swarm_particle[j].update_position(bounds)
            
        A.append(fitness_global_best_particle_position) # record the best fitness
        
        # Visualization
        ax.plot(A, color='r')
        fig.canvas.draw()
        ax.set_xlim(left=max(0, i - iterations), right=i + 3)
        
    print("\nOptimal solution:", global_best_particle_position)
    print("Objective function value:", fitness_global_best_particle_position)
    plt.show()



if __name__ == "__main__":
    bounds = [(-5.12, 5.12), (-5.12, 5.12)]  # upper and lower bounds of variables
    nv = 2  # number of variables
    mm = -1  # if minimization problem, mm = -1; if maximization problem, mm = 1
      
    # THE FOLLOWING PARAMETERS ARE OPTIONAL
    particle_size = 10  # number of particles
    iterations = 100  # max number of iterations
    w = 0.75  # inertia constant
    c1 = 1  # cognitive constant
    c2 = 2  # social constant
    
    optimize_particle_swarm(mm, objective_function, bounds, particle_size, iterations, w, c1, c2)
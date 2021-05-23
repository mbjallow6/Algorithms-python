#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
import math
import time
import numpy as np
import matplotlib.pyplot as plt
from igraph import Graph  
from tqdm import tqdm
from itertools import product
import itertools 
import pandas as pd
import heapq
import copy


# In[2]:


def independent_cascade_model_for_simulated_annealing(graph, seed_set, probability, iteration=1000):
    spread = 0
    for i in range(iteration):
        np.random.seed(i)
        active = seed_set[:]
        new_active = seed_set[:]
        while new_active:
            activated_nodes = []
            for node in new_active:
                neighbors = graph.neighbors(node, mode='out')
                success = np.random.uniform(0, 1, len(neighbors)) < probability
                activated_nodes += list(np.extract(success, neighbors))
            new_active = list(set(activated_nodes) - set(active))
            active=list(set(active))
            active += new_active
        spread += len(active)
    return spread / iteration


# In[3]:


def create_neighbor_solution_set(graph,current_solution):
    tmp1=random.sample(current_solution, 1)
    tmp2=random.sample(list(range(graph.vcount())), 1)
    if len(current_solution) == graph.vcount():
        return current_solution
    while set(tmp2)-set(current_solution) == set():
        tmp2=random.sample(list(range(graph.vcount())), 1)
    return [tmp2[0] if x==tmp1[0] else x for x in current_solution]

    
    
    
def algo(graph, k, probability, iteration):
    start_time = time.time()
    count=0
    initial_tempreture=20
    final_tempreture=0
    alpha=1
    bita=5
    population=list(range(graph.vcount()))
    current_solution=random.sample(population, k)   
    while initial_tempreture > final_tempreture:
        current_spread=independent_cascade_model_for_simulated_annealing(graph,current_solution, probability, iteration)
        count=count+1
        new_solution=create_neighbor_solution_set(graph,current_solution)
        new_spread=independent_cascade_model_for_simulated_annealing(graph,new_solution, probability, iteration)
        energy_change=new_spread-current_spread
        if energy_change>0:
            current_solution=new_solution
        else:
            if math.exp(energy_change/initial_tempreture) > random.uniform(0, 1):
                current_solution=new_solution
                
        if count >= bita:
            count=0
            initial_tempreture=initial_tempreture-alpha
        #print(initial_tempreture)
    end_time=time.time()
    
    spread=independent_cascade_model_for_simulated_annealing(graph, current_solution, probability, iteration)
    return current_solution,round(spread*100/graph.vcount(),2), round(end_time-start_time, 2)
        
    
    





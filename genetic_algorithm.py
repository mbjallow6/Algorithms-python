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


class chromosome:
    encoding=[]
    fitness=0
    generation_count=0
    def __gt__(self, other):
        return self.fitness < other.fitness


# In[3]:


def generate_initial_population(graph,population_size,k):
    initial_population=[]
    for i in range(population_size):
        x=chromosome()
        x.encoding=list(random.sample(list(range(graph.vcount())), k))
        initial_population.append(x)
    return initial_population


# In[4]:


def compute_fitness(graph, seed_nodes, prob, n_iters):
    spread_with_seed_nodes=0
    for i in range(n_iters):
        np.random.seed(i)
        active_nodes=copy.deepcopy(seed_nodes)
        active_nodes=list(set(active_nodes))
        newly_active_nodes=copy.deepcopy(seed_nodes)
        while len(newly_active_nodes)>0:
            activated_nodes=[]
            for node in newly_active_nodes:
                neighbors=graph.neighbors(node,mode='out')
                success=np.random.uniform(0,1,len(neighbors))<prob
                activated_nodes+=list(np.extract(success, neighbors))
            activated_nodes=list(set(activated_nodes))#remove duplicate nodes
            newly_active_nodes=list(set(activated_nodes)-set(active_nodes))
            active_nodes+=newly_active_nodes
        spread_with_seed_nodes+=len(active_nodes)
        #print(i,' spread_with_seed_nodes: ',spread_with_seed_nodes)
    return spread_with_seed_nodes/ n_iters


# In[ ]:





# In[5]:


def selection(population):
    male_chromosome=population[0]
    female_chromosome=max(population[1],population[2])
    return male_chromosome, female_chromosome


# In[6]:


def crossover(chromosome1, chromosome2):
    male_chromosome=copy.deepcopy(chromosome1)
    female_chromosome=copy.deepcopy(chromosome2)
    crossover_point=random.randint(0,len(male_chromosome.encoding)-1)
    for i in range(0,crossover_point,1):
        tmp=male_chromosome.encoding[i]
        male_chromosome.encoding[i]=female_chromosome.encoding[i]
        female_chromosome.encoding[i]=tmp
    male_chromosome.generation_count=male_chromosome.generation_count+1
    female_chromosome.generation_count=female_chromosome.generation_count+1
    return male_chromosome, female_chromosome


# In[7]:


def mutation(graph,population,offspring1,offspring2,mutation_prob,prob, n_iters):
    for candidate in [offspring1,offspring2]:
        if(random.uniform(0, 1)>mutation_prob):
            mutation_point=random.randint(0,len(candidate.encoding)-1)
            #print('mutation_point='+str(mutation_point))
            candidate.encoding[mutation_point]=random.randint(0,graph.vcount()-1)
        candidate.fitness=compute_fitness(graph,candidate.encoding, prob, n_iters)
        heapq.heappush(population,candidate)
    return population


# In[8]:


def algo(graph,generation_count,population_size,k):
    start_time = time.time()
    prob, n_iters=0.1,10
    mutation_prob=0.1
    population=generate_initial_population(graph,population_size,k)
    for i in range(len(population)):
        population[i].fitness=compute_fitness(graph, population[i].encoding, prob, n_iters)
    heapq.heapify(population)
    for i in (range(generation_count)):
        male_chromosome, female_chromosome=selection(population)
        #print( male_chromosome.encoding,'\t\t\t\t',round(male_chromosome.fitness*100/graph.vcount(),2),'\t', male_chromosome.generation_count)
        offspring1,offspring2=crossover(male_chromosome, female_chromosome)
        population=mutation(graph,population,offspring1,offspring2,mutation_prob,prob, n_iters)
    end_time=time.time()
    return population[0].encoding,round(population[0].fitness*100/graph.vcount(),2), round(end_time-start_time, 2)




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
import numpy as np
import networkx as nx
import networkx.algorithms.community as nx_com
import networkx.algorithms.centrality as nx_cen
import random 
import copy
import math
import csv
import matplotlib.pyplot as plt
from random import uniform, seed
import pandas as pd
import time
from collections import Counter



def get_RRS(G,p):   
    """
    Inputs: G:  Ex2 dataframe of directed edges. Columns: ['source','target']
            p:  Disease propagation probability
    Return: A random reverse reachable set expressed as a list of nodes
    """
    
    # Step 1. Select random source node
    source = random.choice(np.unique(G['source']))
    
    # Step 2. Get an instance of g from G by sampling edges  
    g = G.copy().loc[np.random.uniform(0,1,G.shape[0]) < p]

    # Step 3. Construct reverse reachable set of the random source node
    new_nodes, RRS0 = [source], [source]   
    # print('New nodes: ', new_nodes)
    # print('RRS0: ', RRS0)
    # while new_nodes:
    pop = []
    for i in range(len(new_nodes)):
        
        # Limit to edges that flow into the source node
        temp = g.loc[g['target'].isin(new_nodes)]

        # Extract the nodes flowing into the source node
        temp = temp['source'].tolist()
        # print('Extract the nodes flowing into the source node: ',temp)

        # Add new set of in-neighbors to the RRS
        RRS = list(set(RRS0 + temp))
        pop.append(list(set(RRS0 + temp)))
        # print('Add new set of in-neighbors to the RRS: ',RRS)

        # Find what new nodes were added
        new_nodes = list(set(RRS) - set(RRS0))

        # RRS0.append(list(set(RRS0 + temp)))

        # Reset loop variables
        RRS0 = RRS[:]
        # print('RRS0: ', RRS0)

    return(pop)
class chromosome:
    encoding=[]
    # print("Printing Encoding: ", encoding)
    fitness=0
    generation_count=0
    def __gt__(self, other):
        return self.fitness < other.fitness


def get_initial_population(df,population_size,p,k):
    RR =[get_RRS(df,p) for _ in range(k)]
    initial_population=[]
    # print("Printing Population: ", initial_population)
    for i in range(population_size):
        parent = []
        for i in RR: 
            for x in i:   
                gene = Counter(x).most_common()[0][0]
                parent.append(gene)
        x=chromosome()
        x.encoding= parent
        initial_population.append(x)
    return initial_population

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

def selection(population):
    male_chromosome=population[0]
    female_chromosome=max(population[1],population[2])
    return male_chromosome, female_chromosome

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
def mutation(graph,population,offspring1,offspring2,mutation_prob,prob, n_iters):
    for candidate in [offspring1,offspring2]:
        if(random.uniform(0, 1)>mutation_prob):
            mutation_point=random.randint(0,len(candidate.encoding)-1)
            #print('mutation_point='+str(mutation_point))
            candidate.encoding[mutation_point]=random.randint(0,graph.vcount()-1)
        candidate.fitness=compute_fitness(graph,candidate.encoding, prob, n_iters)
        heapq.heappush(population,candidate)
    return population

def algo(graph,df,generation_count,population_size,k):
    start_time = time.time()
    prob, n_iters=0.1,10
    mutation_prob=0.1
    population=get_initial_population(df,population_size,prob,k)
    # print('Population', population)
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
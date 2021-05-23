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


def independent_cascade_model(graph, seed_nodes, prob,n_iters,u):
    
    spread_with_seed_nodes=0
    spread_with_seed_nodes_and_u=0
    for i in range(n_iters):
        np.random.seed(i)
        active_nodes=copy.deepcopy(seed_nodes)
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
        newly_active_nodes=[u]
        while len(newly_active_nodes)>0:
            activated_nodes=[]
            for node in newly_active_nodes:
                neighbors=graph.neighbors(node,mode='out')
                success=np.random.uniform(0,1,len(neighbors))<prob
                activated_nodes+=list(np.extract(success, neighbors))
            activated_nodes=list(set(activated_nodes))#remove duplicate nodes
            newly_active_nodes=list(set(activated_nodes)-set(active_nodes))
            active_nodes+=newly_active_nodes
        spread_with_seed_nodes_and_u+=len(active_nodes)
        #print(i,' spread_with_seed_nodes: ',spread_with_seed_nodes,' spread_with_seed_nodes_and_u: ',spread_with_seed_nodes_and_u)
    return ((spread_with_seed_nodes/n_iters), (spread_with_seed_nodes_and_u / n_iters))


# In[3]:


def algo(graph, k, probability=0.1, iteration=1000):
    start_time = time.time()
    seed_set=[]
    best_vertex=None
    best_marginal_gain=-np.inf
    total_spread=0
    while len(seed_set)<k:
        for u in list(set(range(graph.vcount()))-set(seed_set)):
            x,y=independent_cascade_model(graph, seed_set, probability, iteration,u)#-independent_cascade_model(graph, seed_set, probability, iteration)
            current_marginal_gain=y-x
            if current_marginal_gain>best_marginal_gain:
                best_marginal_gain=current_marginal_gain
                best_vertex=u
                total_spread=y
        seed_set=seed_set+[best_vertex]
        best_vertex=None
        best_marginal_gain=-np.inf
    end_time=time.time()
    #spread=independent_cascade_model(graph, seed_set, probability, iteration)
    x,y=independent_cascade_model(graph, seed_set, probability, iteration,u)#-independent_cascade_model(graph, seed_set, probability, iteration)
            
    return seed_set, round(y*100/graph.vcount(),2), round(end_time-start_time, 2)



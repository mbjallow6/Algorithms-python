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


def independent_cascade_model_for_celfpp(graph, seed_nodes, prob,n_iters,prev_best,u):
       
    spread_with_seed_nodes=0
    spread_with_seed_nodes_and_u=0
    spread_with_seed_nodes_and_prev_best=0
    spread_with_seed_nodes_and_prev_best_and_u=0
    
    # simulate the spread process over multiple runs
    for i in range(n_iters):
        np.random.seed(i)
        #spread_with_seed_nodes#
        ##############################################
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
        active_nodes_due_to_seed_nodes=copy.deepcopy(active_nodes)
        #################################################
        
        #spread_with_seed_nodes_and_u#
        ##################################################
        active_nodes=copy.deepcopy(active_nodes_due_to_seed_nodes)
        newly_active_nodes=newly_active_nodes=[u]
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
        #########################################################
        
        #spread_with_seed_nodes_and_prev_best#
        ###########################################################
        #spread_with_seed_nodes_and_prev_best+=len(normal_active)
        active_nodes=copy.deepcopy(active_nodes_due_to_seed_nodes)
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
        spread_with_seed_nodes_and_prev_best+=len(active_nodes)
        active_nodes_due_to_seed_nodes_and_prev_best=copy.deepcopy(active_nodes)
        ################################################################
        
        #spread_with_seed_nodes_and_prev_best_and_u#
        ################################################################
        active_nodes=copy.deepcopy(active_nodes_due_to_seed_nodes_and_prev_best)
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
        spread_with_seed_nodes_and_prev_best_and_u+=len(active_nodes)
        ###################################################################
    return (spread_with_seed_nodes / n_iters),(spread_with_seed_nodes_and_u/n_iters),(spread_with_seed_nodes_and_prev_best/n_iters),(spread_with_seed_nodes_and_prev_best_and_u/n_iters)


# In[3]:


import heapq
class celfpp_node:
    def __init__(self, mg1,mg2,prev_best,flag):
        self.mg1=mg1
        self.mg2=mg2
        self.prev_best=prev_best
        self.flag=flag
    def __gt__(self, other):
        return self.mg1 > other.mg1

def algo(graph, k, probability, iteration=1000):
    start_time = time.time()
    seed_set=[]
    queue=[]
    last_seed=None
    current_best=None
    total_spread=0
    for u in (range(graph.vcount())):
        w,x,y,z=independent_cascade_model_for_celfpp(graph, [], probability, iteration,u,current_best)
        #print('normal_spread: ',w,' spread_with_u: ',x,' spread_with_prev_best: ',y,' spread_with_u_and_prev_best: ',z)
        mg1=x-w
        prev_best=current_best
        if current_best:
            #mg2=independent_cascade_model(graph, [u], probability,iteration,)
            mg2=z-y
        else:
            mg2=mg1
        flag=0
        heapq.heappush(queue,(-mg1,u,celfpp_node(mg1,mg2,prev_best,flag))) 
        if current_best and current_best_marginal_gain > mg1:
            current_best = current_best 
        else :
            current_best = u
            current_best_marginal_gain=mg1

        #print(len(queue))
    while len(seed_set)<k:
        #print(len(seed_set))
        sp,vertex_number,u=heapq.heappop(queue)
        if u.flag==len(seed_set):
            seed_set=seed_set+[vertex_number]
            last_seed=vertex_number
            
            continue
        else:
            if u.prev_best==last_seed:
                u.mg1=u.mg2
            else:
                w,x,y,z=independent_cascade_model_for_celfpp(graph, seed_set, probability, iteration,u.prev_best,vertex_number)
                u.mg1=x-w #independent_cascade_model(graph, seed_set+[vertex_number], probability,iteration)-independent_cascade_model(graph, seed_set, probability, iteration)
                u.prev_best=current_best
                total_spread=x
                if current_best!=vertex_number:
                    u.mg2=z-y#independent_cascade_model(graph, seed_set+[current_best,vertex_number],probability,iteration)-independent_cascade_model(graph, seed_set+[current_best], probability,iteration)
                else:
                    u.mg2=u.mg1
                #print('normal_spread: ',w,' spread_with_u: ',x,' spread_with_prev_best: ',y,' spread_with_u_and_prev_best: ',z)

        u.flag=len(seed_set)
        if current_best and current_best_marginal_gain < u.mg1:
            current_best = vertex_number
            current_best_marginal_gain=u.mg1
        heapq.heappush(queue,(-u.mg1,vertex_number,u))
        #current_best=queue[0][1]

    end_time=time.time()
    w,x,y,z=independent_cascade_model_for_celfpp(graph, seed_set, probability, iteration,u.prev_best,vertex_number)
    return seed_set,round(w*100/graph.vcount(),2), round(end_time-start_time, 2)
    








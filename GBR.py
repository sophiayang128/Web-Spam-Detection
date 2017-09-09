
# coding: utf-8

# In[1]:

from __future__ import division
import networkx as nx
import math
from scipy.sparse.linalg.eigen.arpack import eigsh as largest_eigsh
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv


# In[2]:

def GoodBadRank(G,good_seeds,bad_seeds):
    good_score = []
    bad_score = []
    
    #initilization
    for p in G.nodes():
        if p in good_seeds:
            good_score.append(1/len(good_seeds))
        else:
            good_score.append(0)

    for p in G.nodes():    
        if p in bad_seeds:
            bad_score.append(1/len(bad_seeds))
        else:
            bad_score.append(0)
   
    #iterate until convergence
    i = 0
    while True:
        i = i+1
        #print i
        pre_good_score = list(good_score)
        pre_bad_score = list(bad_score)
        
        for p in G.nodes():

            #GoodRank score
            
            in_degree_list = G.predecessors(p)
            good_sum_list = []
            for q in in_degree_list:       
                if good_score[q] is not None and bad_score[q] is not None:
                    out_q = G.out_degree(q)
                    if out_q != 0:
                        g1 = good_score[q]/out_q
                    else:
                        g1 = good_score[q]
                    if good_score[q] == 0 and bad_score[q] == 0:
                        g = g1 
                    else:
                        p_g = good_score[q]/(good_score[q] + bad_score[q])
                        g = g1*p_g
                    good_sum_list.append(g)
            good_sum = sum(good_sum_list)
            if p in good_seeds:
                d = 1/len(good_seeds)
            else:
                d = 0
            good_score[p] = 0.85*good_sum + 0.15*d
           
            #BadRank score
            
            out_degree_list = G.successors(p)
            bad_sum_list = []
            for q in out_degree_list:                
                if good_score[q] is not None and bad_score[q] is not None:
                    in_q = G.in_degree(q)
                    if in_q != 0:
                        b1 = bad_score[q]/in_q
                    else:
                        b1 = bad_score[q]
                    if good_score[q] == 0 and bad_score[q] == 0:
                        b = b1
                    else:
                        p_b = bad_score[q]/(good_score[q] + bad_score[q])
                        b = b1*p_b     
                    bad_sum_list.append(b)
            bad_sum = sum(bad_sum_list)
            if p in bad_seeds:
                d = 1/len(bad_seeds)
            else:
                d = 0
            bad_score[p] = 0.85*bad_sum + 0.15*d
       
        print "good_score", good_score[:50]
        print "pre_good_score", pre_good_score[:50]
        
        
        #if pre_good_score == good_score and pre_bad_score == bad_score:
        if i == 40:
            break
    return good_score, bad_score   


# In[ ]:




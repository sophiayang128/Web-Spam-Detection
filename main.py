
# coding: utf-8

# In[685]:

import urllib
import networkx as nx
import matplotlib.pyplot as plt 
import numpy as np 
import urllib2
import requests
from lxml import etree
import re  
import chardet    
import random
import igraph
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn import preprocessing
from sklearn import ensemble
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier  
from sklearn import cross_validation, metrics  
from sklearn import grid_search
from sklearn.grid_search import GridSearchCV 
import pandas as pd
import os
from datetime import datetime
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import nltk
import csv
from __future__ import division
import networkx as nx
import math
from scipy.sparse.linalg.eigen.arpack import eigsh as largest_eigsh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from GBR import *


# In[839]:

#extract host_id
with open('uk-2007-05.content_based_features.csv', 'rb') as csvfile:
    content_2007 = csv.reader(csvfile.read().splitlines())
    host_id = [row[0] for row in content_2007]
csvfile.close()


# In[688]:

#read graph information from file 
with open("uk-2007-05.hostgraph_weighted.graph-txt") as f:
    reader = csv.reader(f,delimiter=' ')
    i = 0
    edge_dict = {}
    reader.next()
    for row in reader:
        dest_list = []
        for element in row:
            dest = element.split(":")[0]
            dest_list.append(dest)
        edge_dict[i] = dest_list
        i = i + 1
f.close()

#contruct the graph
G = nx.DiGraph()
node_list = range(114528)
G.add_nodes_from(node_list)
edge_list = []
for key in edge_dict:
    for element in edge_dict[key]:
        edge_list.append((key,int(element)))
G.add_edges_from(edge_list)


# In[798]:

###################### establish training set ########################
with open('WEBSPAM-UK2007-SET1-labels.txt','r') as txtfile:
    label_file = txtfile.readlines()

id_label = []
allwords = []
label = [] 

for i in range(len(label_file)):  
    allwords.append(label_file[i].split(' '))

for i in range(len(allwords)):
    id_label.append(allwords[i][0])
    label.append(allwords[i][1])

for i in range(len(id_label)):
    if label[i]=='spam':
        label[i]=1
    elif label[i]=='nonspam':
        label[i]=0
    else:
        label[i]=2  #undecided
    if id_label[i] not in host_id:   #not in content file
        id_label[i]=0
        label[i]=3
      
    
#delete content not in the file
dellist = []
delid = []

for i in range(len(label)):
    if label[i]==3:
        dellist.append(label[i])
        delid.append(id_label[i])
for i in dellist:
    label.remove(i)
for i in delid:
    id_label.remove(i)
    

#delete undecided pages
dellist = []
delid = []

for i in range(len(label)):
    if label[i]==2:
        dellist.append(label[i])
        delid.append(id_label[i])
for i in dellist:
    label.remove(i)
for i in delid:
    id_label.remove(i)


# In[693]:

'''
split to majority(nonspam list) and minority(spam list)
input: label
output: nonspam_list, spam_list
'''
def split_spam_nonspam(label):
    nonspam_list = []
    spam_list = []
    for i in range(len(label)):
        if label[i] == 0:
            nonspam_list.append(i)
        if label[i] == 1:
            spam_list.append(i)
    return nonspam_list,spam_list


# In[745]:

##################### link-based features ####################

######### Goodbadrank features ##########
#select seeds for Goodbadrank feature
good_seeds = []
bad_seeds = []
with open("WEBSPAM-UK2007-SET1-labels.txt") as f:
    reader = csv.reader(f, delimiter=' ')
    for row in reader:
        if row[1] == "nonspam":
            good_seeds.append(int(row[0]))
        if row[1] == "spam":
            bad_seeds.append(int(row[0]))
f.close()

import random
good_seeds = random.sample(good_seeds,2000)
bad_seeds = random.sample(bad_seeds,100)


# In[746]:

#invoke GoodBadRank function in GBR.py
g,b = GoodBadRank(G,good_seeds,bad_seeds)

#for training set
t_g = []
t_b = []
for ids in id_label:
    t_g.append(g[int(ids)])
    t_b.append(b[int(ids)])


# In[694]:

######### extract other graph-based features #########

#Log of indegree of home page (=hp) of the host
log_indegree_hp = []
#Log of indegree of page with maximum pagerank (=mp) of the host
log_indegree_mp = []
#Log of outdegree of hp
log_outdegree_hp = []
#Log of outdegree of mp
log_outdegree_mp = []
#Edge-reciprocity (fraction of out-links that are in-links) of hp
reciprocity_hp = []
#Edge-reciprocity (fraction of out-links that are in-links) of mp
reciprocity_mp = []
#Log of assortativity (my degree divided by average degree of neighbors) of hp
log_assortativity_hp = []
#Log of assortativity (my degree divided by average degree of neighbors) of mp
log_assortativity_mp = []
#Log of average indegree of outneighbors of hp
log_avgin_of_out_hp = []
#Log of average indegree of outneighbors of mp
log_avgin_of_out_mp = []
#Log of sum of the indegree of outneighbors of hp
log_avgin_of_out_hp_times_outdegree_hp = []
#Log of sum of the indegree of outneighbors of mp
log_avgin_of_out_hp_times_outdegree_mp = []
#Log of average outdegree of inneighbors of hp
log_avgout_of_in_hp = []
#Log of average outdegree of inneighbors of mp
log_avgout_of_in_mp = []
#Log of the sum of the outdegree of inneighbors of hp
log_avgout_of_in_hp_times_indegree_hp = []
#Log of the sum of the outdegree of inneighbors of mp
log_avgout_of_in_hp_times_indegree_mp = []

eq_hp_mp = []
#Is the homepage the same as the page with max. pagerank? 0=no/1=yes

log_pagerank_hp = []
#Log of pagerank of hp

log_pagerank_mp = []
#Log of pagerank of mp

log_indegree_hp_divided_pagerank_hp = []
#Log of indegree/pagerank of hp

log_indegree_mp_divided_pagerank_mp = []
#Log of indegree/pagerank of mp

log_outdegree_hp_divided_pagerank_hp = []
#Log of outdegree/pagerank of hp

log_outdegree_mp_divided_pagerank_mp = []
#Log of outdegree/pagerank of mp

log_prsigma_hp = []
#Log of st. dev of PageRank of inneighbors of hp

log_prsigma_mp = []
#Log of st. dev of PageRank of inneighbors of mp

log_prsigma_hp_divided_pagerank_hp = []
#Log of st. dev of PageRank of inneighbors / pagerank of hp

log_prsigma_mp_divided_pagerank_mp = []
#Log of st. dev of PageRank of inneighbors / pagerank of mp

pagerank_mp_divided_pagerank_hp = []
#PageRank of mp divided by PageRank of hp

log_trustrank_hp = []
#Log of TrustRank (using 3,800 trusted nodes from ODP) of hp

log_trustrank_mp = []
#Log of TrustRank (using 3,800 trusted nodes from ODP) of mp

log_trustrank_hp_divided_pagerank_hp = []
#Log of TrustRank/PageRank of hp

log_trustrank_mp_divided_pagerank_mp = []
#Log of TrustRank/PageRank of mp

log_trustrank_hp_divided_indegree_hp = []
#Log of TrustRank/indegree of hp

log_trustrank_mp_divided_indegree_mp = []
#Log of TrustRank/indegree of mp

trustrank_mp_divided_trustrank_hp = []
#TrustRank of hp divided by TrustRank of mp

log_siteneighbors_1_hp = []
#Log of number of different supporters (different sites) at distance 1 from hp

log_siteneighbors_2_hp = []
#Log of number of different supporters (different sites) at distance 2 from hp

log_siteneighbors_3_hp = []
#Log of number of different supporters (different sites) at distance 3 from hp

log_siteneighbors_4_hp = []
#Log of number of different supporters (different sites) at distance 4 from hp

log_siteneighbors_1_mp = []
#Log of number of different supporters (different sites) at distance 1 from mp

log_siteneighbors_2_mp = []
#Log of number of different supporters (different sites) at distance 2 from mp

log_siteneighbors_3_mp = []
#Log of number of different supporters (different sites) at distance 3 from mp

log_siteneighbors_4_mp = []
#Log of number of different supporters (different sites) at distance 4 from mp

log_siteneighbors_1_hp_divided_pagerank_hp = []
#Log of number of different supporters (different sites) at distance 1 from hp divided by PageRank

log_siteneighbors_2_hp_divided_pagerank_hp = []
#Log of number of different supporters (different sites) at distance 2 from hp divided by PageRank

log_siteneighbors_3_hp_divided_pagerank_hp = []
#Log of number of different supporters (different sites) at distance 3 from hp divided by PageRank

log_siteneighbors_4_hp_divided_pagerank_hp = []
#Log of number of different supporters (different sites) at distance 4 from hp divided by PageRank

log_siteneighbors_1_mp_divided_pagerank_mp = []
#Log of number of different supporters (different sites) at distance 1 from mp divided by PageRank

log_siteneighbors_2_mp_divided_pagerank_mp = []
#Log of number of different supporters (different sites) at distance 2 from mp divided by PageRank

log_siteneighbors_3_mp_divided_pagerank_mp = []
#Log of number of different supporters (different sites) at distance 3 from mp divided by PageRank

log_siteneighbors_4_mp_divided_pagerank_mp = []
#Log of number of different supporters (different sites) at distance 4 from mp divided by PageRank

log_siteneighbors_4_hp_divided_siteneighbors_3_hp = []
#Log of number of different supporters (different sites) at distance 4 from hp divided by number of different supporters (different sites) at distance 3 from hp

log_siteneighbors_4_mp_divided_siteneighbors_3_mp = []
#Log of number of different supporters (different sites) at distance 4 from mp divided by number of different supporters (different sites) at distance 3 from mp

log_siteneighbors_3_hp_divided_siteneighbors_2_hp = []
#Log of number of different supporters (different sites) at distance 3 from hp divided by number of different supporters (different sites) at distance 2 from hp

log_siteneighbors_3_mp_divided_siteneighbors_2_mp = []
#Log of number of different supporters (different sites) at distance 3 from mp divided by number of different supporters (different sites) at distance 2 from mp

log_siteneighbors_2_hp_divided_siteneighbors_1_hp = []
#Log of number of different supporters (different sites) at distance 2 from hp divided by number of different supporters (different sites) at distance 1 from hp

log_siteneighbors_2_mp_divided_siteneighbors_1_mp = []
#Log of number of different supporters (different sites) at distance 2 from mp divided by number of different supporters (different sites) at distance 1 from mp

log_min_siteneighbors_hp = []
#Log of minimum change in the number of supporters (different sites) at distance i over distance i-1, for i=2..4, hp

log_min_siteneighbors_mp = []
#Log of minimum change in the number of supporters (different sites) at distance i over distance i-1, for i=2..4, mp

log_max_siteneighbors_hp = []
#Log of maximum change in the number of supporters (different sites) at distance i over distance i-1, for i=2..4, hp

log_max_siteneighbors_mp = []
#Log of maximum change in the number of supporters (different sites) at distance i over distance i-1, for i=2..4, mp

log_avg_siteneighbors_hp = []
#Log of average change in the number of supporters (different sites) at distance i over distance i-1, for i=2..4, hp

log_avg_siteneighbors_mp = []
#Log of average change in the number of supporters (different sites) at distance i over distance i-1, for i=2..4, mp

log_siteneighbors_4_hp_siteneighbors_3_hp_pagerank_hp = []
#Log of supporters at distance exactly 4 (different sites) divided by PageRank, hp

log_siteneighbors_4_hp_siteneighbors_3_mp_pagerank_mp = []
#Log of supporters at distance exactly 4 (different sites) divided by PageRank, mp

log_siteneighbors_3_hp_siteneighbors_2_hp_pagerank_hp = []
#Log of supporters at distance exactly 3 (different sites) divided by PageRank, hp

log_siteneighbors_3_hp_siteneighbors_2_mp_pagerank_mp = []
#Log of supporters at distance exactly 3 (different sites) divided by PageRank, mp

log_siteneighbors_2_hp_siteneighbors_1_hp_pagerank_hp = []
#Log of supporters at distance exactly 2 (different sites) divided by PageRank, hp

log_siteneighbors_2_hp_siteneighbors_1_hp_pagerank_mp = []
#Log of supporters at distance exactly 2 (different sites) divided by PageRank, mp

siteneighbors_1_hp_divided_siteneighbors_1_mp = []
#Supporters at distance 1 (different sites) of hp over mp

siteneighbors_2_hp_divided_siteneighbors_2_mp = []
#Supporters at distance 2 (different sites) of hp over mp

siteneighbors_3_hp_divided_siteneighbors_3_mp = []
#Supporters at distance 3 (different sites) of hp over mp

siteneighbors_4_hp_divided_siteneighbors_4_mp = []
#Supporters at distance 4 (different sites) of hp over mp

log_neighbors_2_hp = []
#Log of supporters at distance 2, hp (note: supporters at distance 1 is indegree)

log_neighbors_3_hp = []
#Log of supporters at distance 3, hp

log_neighbors_4_hp = []
#Log of supporters at distance 4, hp

log_neighbors_2_mp = []
#Log of supporters at distance 2, mp

log_neighbors_3_mp = []
#Log of supporters at distance 3, mp

log_neighbors_4_mp = []
#Log of supporters at distance 4, mp

log_neighbors_2_hp_divided_pagerank_hp = []
#Log of supporters at distance 2 divided by PageRank, hp

log_neighbors_3_hp_divided_pagerank_hp = []
#Log of supporters at distance 3 divided by PageRank, hp

log_neighbors_4_hp_divided_pagerank_hp = []
#Log of supporters at distance 4 divided by PageRank, hp

log_neighbors_2_mp_divided_pagerank_mp = []
#Log of supporters at distance 2 divided by PageRank, mp

log_neighbors_3_mp_divided_pagerank_mp = []
#Log of supporters at distance 3 divided by PageRank, mp

log_neighbors_4_mp_divided_pagerank_mp = []
#Log of supporters at distance 4 divided by PageRank, mp

log_neighbors_4_hp_divided_neighbors_3_hp = []
#Log of supporters at distance 4 divided by supporters at distance 3, hp

log_neighbors_4_mp_divided_neighbors_3_mp = []
#Log of supporters at distance 4 divided by supporters at distance 3, mp

log_neighbors_3_hp_divided_neighbors_2_hp = []
#Log of supporters at distance 3 divided by supporters at distance 2, hp

log_neighbors_3_mp_divided_neighbors_2_mp = []
#Log of supporters at distance 3 divided by supporters at distance 2, mp

log_neighbors_2_hp_divided_indegree_hp = []
#Log of supporters at distance 2 divided by supporters at distance 1, hp

log_neighbors_2_mp_divided_indegree_mp = []
#Log of supporters at distance 2 divided by supporters at distance 1, mp

log_min_neighbors_hp = []
#Log of minimum change of number of supporters at distance i over supporters at distance i-1, i=2..4, hp

log_min_neighbors_mp = []
#Log of minimum change of number of supporters at distance i over supporters at distance i-1, i=2..4, mp

log_max_neighbors_hp = []
#Log of maximum change of number of supporters at distance i over supporters at distance i-1, i=2..4, hp

log_max_neighbors_mp = []
#Log of maximum change of number of supporters at distance i over supporters at distance i-1, i=2..4, mp

log_avg_neighbors_hp = []
#Log of average change of number of supporters at distance i over supporters at distance i-1, i=2..4, hp

log_avg_neighbors_mp = []
#Log of average change of number of supporters at distance i over supporters at distance i-1, i=2..4, mp

log_neighbors_4_divided_pagerank_hp = []
#Log of number of supporters at distance exactly 4 over pagerank, hp

log_neighbors_4_divided_pagerank_mp = []
#Log of number of supporters at distance exactly 4 over pagerank, mp

log_neighbors_3_divided_pagerank_hp = []
#Log of number of supporters at distance exactly 3 over pagerank, hp

log_neighbors_3_divided_pagerank_mp = []
#Log of number of supporters at distance exactly 3 over pagerank, mp

log_neighbors_2_divided_pagerank_hp = []
#Log of number of supporters at distance exactly 2 over pagerank, hp

log_neighbors_2_divided_pagerank_mp = []
#Log of number of supporters at distance exactly 2 over pagerank, mp

neighbors_2_hp_divided_neighbors_2_mp = []
#Supporters at 2 in hp divided by supporters at 2 in mp

neighbors_3_hp_divided_neighbors_3_mp = []
#Supporters at 3 in hp divided by supporters at 4 in mp

neighbors_4_hp_divided_neighbors_4_mp = []
#Supporters at 3 in hp divided by supporters at 4 in mp

log_truncatedpagerank_1_hp = []
#Log of TruncatedPageRank with T=1, hp

log_truncatedpagerank_2_hp = []
#Log of TruncatedPageRank with T=2, hp

log_truncatedpagerank_3_hp = []
#Log of TruncatedPageRank with T=3, hp

log_truncatedpagerank_4_hp = []
#Log of TruncatedPageRank with T=4, hp

log_truncatedpagerank_1_mp = []
#Log of TruncatedPageRank with T=1, mp

log_truncatedpagerank_2_mp = []
#Log of TruncatedPageRank with T=2, mp

log_truncatedpagerank_3_mp = []
#Log of TruncatedPageRank with T=3, mp

log_truncatedpagerank_4_mp = []
#Log of TruncatedPageRank with T=4, mp

log_truncatedpagerank_1_hp_divided_pagerank_hp = []
#Log of TruncatedPageRank with T=1 divided by PageRank, hp

log_truncatedpagerank_2_hp_divided_pagerank_hp = []
#Log of TruncatedPageRank with T=2 divided by PageRank, hp

log_truncatedpagerank_3_hp_divided_pagerank_hp = []
#Log of TruncatedPageRank with T=3 divided by PageRank, hp

log_truncatedpagerank_4_hp_divided_pagerank_hp = []
#Log of TruncatedPageRank with T=4 divided by PageRank, hp

log_truncatedpagerank_1_mp_divided_pagerank_mp = []
#Log of TruncatedPageRank with T=1 divided by PageRank, mp

log_truncatedpagerank_2_mp_divided_pagerank_mp = []
#Log of TruncatedPageRank with T=2 divided by PageRank, mp

log_truncatedpagerank_3_mp_divided_pagerank_mp = []
#Log of TruncatedPageRank with T=3 divided by PageRank, mp

log_truncatedpagerank_4_mp_divided_pagerank_mp = []
#Log of TruncatedPageRank with T=4 divided by PageRank, mp

truncatedpagerank_4_hp_divided_truncatedpagerank_3_hp = []
#Log of TruncatedPageRank with T=4 divided by TruncatedPageRank with T=3, hp

truncatedpagerank_4_mp_divided_truncatedpagerank_3_mp = []
#Log of TruncatedPageRank with T=4 divided by TruncatedPageRank with T=3, mp

truncatedpagerank_3_hp_divided_truncatedpagerank_2_hp = []
#Log of TruncatedPageRank with T=3 divided by TruncatedPageRank with T=3, hp

truncatedpagerank_3_mp_divided_truncatedpagerank_2_mp = []
#Log of TruncatedPageRank with T=3 divided by TruncatedPageRank with T=3, mp

truncatedpagerank_2_hp_divided_truncatedpagerank_1_hp = []
#Log of TruncatedPageRank with T=2 divided by TruncatedPageRank with T=3, hp

truncatedpagerank_2_mp_divided_truncatedpagerank_1_mp = []
#Log of TruncatedPageRank with T=2 divided by TruncatedPageRank with T=3, mp

log_min_truncatedpagerank_hp = []
#Log of minimum of TruncatedPageRank with T=i over TruncatedPageRank with T=i-1, i=2..4, hp

log_min_truncatedpagerank_mp = []
#Log of minimum of TruncatedPageRank with T=i over TruncatedPageRank with T=i-1, i=2..4, mp

log_max_truncatedpagerank_hp = []
#Log of maximum of TruncatedPageRank with T=i over TruncatedPageRank with T=i-1, i=2..4, hp

log_max_truncatedpagerank_mp = []
#Log of maximum of TruncatedPageRank with T=i over TruncatedPageRank with T=i-1, i=2..4, mp

log_avg_truncatedpagerank_hp = []
#Log of average of TruncatedPageRank with T=i over TruncatedPageRank with T=i-1, i=2..4, hp

log_avg_truncatedpagerank_mp = []
#Log of average of TruncatedPageRank with T=i over TruncatedPageRank with T=i-1, i=2..4, mp

truncatedpagerank_1_mp_divided_truncatedpagerank_1_hp = []
#TruncatedPageRank with T=1 in hp, divided by TruncatedPageRank with T=1 in mp

truncatedpagerank_2_mp_divided_truncatedpagerank_2_hp = []
#TruncatedPageRank with T=2 in hp, divided by TruncatedPageRank with T=2 in mp

truncatedpagerank_3_mp_divided_truncatedpagerank_3_hp = []
#TruncatedPageRank with T=3 in hp, divided by TruncatedPageRank with T=3 in mp

truncatedpagerank_4_mp_divided_truncatedpagerank_4_hp = []
#TruncatedPageRank with T=4 in hp, divided by TruncatedPageRank with T=4 in mp

 
# open file
with open('uk-2007-05.link_based_features_transformed.csv', 'rb') as f:
    reader = csv.reader(f)
 
    # read file row by row
    rowNr = 0
    for row in reader:
        # Skip the header row.
        if rowNr >= 1:
           
            log_indegree_hp.append(row[1])
            log_indegree_mp.append(row[2])
            log_outdegree_hp.append(row[3])
            log_outdegree_mp.append(row[4])
            reciprocity_hp.append(row[5])
#Edge-reciprocity (fraction of out-links that are in-links) of mp
            reciprocity_mp.append(row[6])
#Log of assortativity (my degree divided by average degree of neighbors) of hp
            log_assortativity_hp.append(row[7])
#Log of assortativity (my degree divided by average degree of neighbors) of mp
            log_assortativity_mp.append(row[8])
#Log of average indegree of outneighbors of hp
            log_avgin_of_out_hp.append(row[9])
#Log of average indegree of outneighbors of mp
            log_avgin_of_out_mp.append(row[10])
#Log of sum of the indegree of outneighbors of hp
            log_avgin_of_out_hp_times_outdegree_hp.append(row[11])
#Log of sum of the indegree of outneighbors of mp
            log_avgin_of_out_hp_times_outdegree_mp.append(row[12])
#Log of average outdegree of inneighbors of hp
            log_avgout_of_in_hp.append(row[13])
#Log of average outdegree of inneighbors of mp
            log_avgout_of_in_mp.append(row[14])
#Log of the sum of the outdegree of inneighbors of hp
            log_avgout_of_in_hp_times_indegree_hp.append(row[15])
#Log of the sum of the outdegree of inneighbors of mp
            log_avgout_of_in_hp_times_indegree_mp.append(row[16])

            eq_hp_mp.append(row[17])
#Is the homepage the same as the page with max. pagerank? 0=no/1=yes

            log_pagerank_hp.append(row[18])
#Log of pagerank of hp

            log_pagerank_mp.append(row[19])
#Log of pagerank of mp

            log_indegree_hp_divided_pagerank_hp.append(row[20])
#Log of indegree/pagerank of hp

            log_indegree_mp_divided_pagerank_mp.append(row[21])
#Log of indegree/pagerank of mp

            log_outdegree_hp_divided_pagerank_hp.append(row[22])
#Log of outdegree/pagerank of hp

            log_outdegree_mp_divided_pagerank_mp.append(row[23])
#Log of outdegree/pagerank of mp

            log_prsigma_hp.append(row[24])
#Log of st. dev of PageRank of inneighbors of hp

            log_prsigma_mp.append(row[25])
#Log of st. dev of PageRank of inneighbors of mp

            log_prsigma_hp_divided_pagerank_hp.append(row[26])
#Log of st. dev of PageRank of inneighbors / pagerank of hp

            log_prsigma_mp_divided_pagerank_mp.append(row[27])
#Log of st. dev of PageRank of inneighbors / pagerank of mp

            pagerank_mp_divided_pagerank_hp.append(row[28])
#PageRank of mp divided by PageRank of hp

            log_trustrank_hp.append(row[29])
#Log of TrustRank (using 3,800 trusted nodes from ODP) of hp

            log_trustrank_mp.append(row[30])
#Log of TrustRank (using 3,800 trusted nodes from ODP) of mp

            log_trustrank_hp_divided_pagerank_hp.append(row[31])
#Log of TrustRank/PageRank of hp

            log_trustrank_mp_divided_pagerank_mp.append(row[32])
#Log of TrustRank/PageRank of mp

            log_trustrank_hp_divided_indegree_hp.append(row[33])
#Log of TrustRank/indegree of hp

            log_trustrank_mp_divided_indegree_mp.append(row[34])
#Log of TrustRank/indegree of mp

            trustrank_mp_divided_trustrank_hp.append(row[35])
#TrustRank of hp divided by TrustRank of mp

            log_siteneighbors_1_hp.append(row[36])
#Log of number of different supporters (different sites) at distance 1 from hp

            log_siteneighbors_2_hp.append(row[37])
#Log of number of different supporters (different sites) at distance 2 from hp

            log_siteneighbors_3_hp.append(row[38])
#Log of number of different supporters (different sites) at distance 3 from hp

            log_siteneighbors_4_hp.append(row[39])
#Log of number of different supporters (different sites) at distance 4 from hp

            log_siteneighbors_1_mp.append(row[40])
#Log of number of different supporters (different sites) at distance 1 from mp

            log_siteneighbors_2_mp.append(row[41])
#Log of number of different supporters (different sites) at distance 2 from mp

            log_siteneighbors_3_mp.append(row[42])
#Log of number of different supporters (different sites) at distance 3 from mp

            log_siteneighbors_4_mp.append(row[43])
#Log of number of different supporters (different sites) at distance 4 from mp

            log_siteneighbors_1_hp_divided_pagerank_hp.append(row[44])
#Log of number of different supporters (different sites) at distance 1 from hp divided by PageRank

            log_siteneighbors_2_hp_divided_pagerank_hp.append(row[45])
#Log of number of different supporters (different sites) at distance 2 from hp divided by PageRank

            log_siteneighbors_3_hp_divided_pagerank_hp.append(row[46])
#Log of number of different supporters (different sites) at distance 3 from hp divided by PageRank

            log_siteneighbors_4_hp_divided_pagerank_hp.append(row[47])
#Log of number of different supporters (different sites) at distance 4 from hp divided by PageRank

            log_siteneighbors_1_mp_divided_pagerank_mp.append(row[48])
#Log of number of different supporters (different sites) at distance 1 from mp divided by PageRank

            log_siteneighbors_2_mp_divided_pagerank_mp.append(row[49])
#Log of number of different supporters (different sites) at distance 2 from mp divided by PageRank

            log_siteneighbors_3_mp_divided_pagerank_mp.append(row[50])
#Log of number of different supporters (different sites) at distance 3 from mp divided by PageRank

            log_siteneighbors_4_mp_divided_pagerank_mp.append(row[51])
#Log of number of different supporters (different sites) at distance 4 from mp divided by PageRank

            log_siteneighbors_4_hp_divided_siteneighbors_3_hp.append(row[52])
#Log of number of different supporters (different sites) at distance 4 from hp divided by number of different supporters (different sites) at distance 3 from hp

            log_siteneighbors_4_mp_divided_siteneighbors_3_mp.append(row[53])
#Log of number of different supporters (different sites) at distance 4 from mp divided by number of different supporters (different sites) at distance 3 from mp

            log_siteneighbors_3_hp_divided_siteneighbors_2_hp.append(row[54])
#Log of number of different supporters (different sites) at distance 3 from hp divided by number of different supporters (different sites) at distance 2 from hp

            log_siteneighbors_3_mp_divided_siteneighbors_2_mp.append(row[55])
#Log of number of different supporters (different sites) at distance 3 from mp divided by number of different supporters (different sites) at distance 2 from mp

            log_siteneighbors_2_hp_divided_siteneighbors_1_hp.append(row[56])
#Log of number of different supporters (different sites) at distance 2 from hp divided by number of different supporters (different sites) at distance 1 from hp

            log_siteneighbors_2_mp_divided_siteneighbors_1_mp.append(row[57])
#Log of number of different supporters (different sites) at distance 2 from mp divided by number of different supporters (different sites) at distance 1 from mp

            log_min_siteneighbors_hp.append(row[58])
#Log of minimum change in the number of supporters (different sites) at distance i over distance i-1, for i=2..4, hp

            log_min_siteneighbors_mp.append(row[59])
#Log of minimum change in the number of supporters (different sites) at distance i over distance i-1, for i=2..4, mp

            log_max_siteneighbors_hp.append(row[60])
#Log of maximum change in the number of supporters (different sites) at distance i over distance i-1, for i=2..4, hp

            log_max_siteneighbors_mp.append(row[61])
#Log of maximum change in the number of supporters (different sites) at distance i over distance i-1, for i=2..4, mp

            log_avg_siteneighbors_hp.append(row[62])
#Log of average change in the number of supporters (different sites) at distance i over distance i-1, for i=2..4, hp

            log_avg_siteneighbors_mp.append(row[63])
#Log of average change in the number of supporters (different sites) at distance i over distance i-1, for i=2..4, mp

            log_siteneighbors_4_hp_siteneighbors_3_hp_pagerank_hp.append(row[64])
#Log of supporters at distance exactly 4 (different sites) divided by PageRank, hp

            log_siteneighbors_4_hp_siteneighbors_3_mp_pagerank_mp.append(row[65])
#Log of supporters at distance exactly 4 (different sites) divided by PageRank, mp

            log_siteneighbors_3_hp_siteneighbors_2_hp_pagerank_hp.append(row[66])
#Log of supporters at distance exactly 3 (different sites) divided by PageRank, hp

            log_siteneighbors_3_hp_siteneighbors_2_mp_pagerank_mp.append(row[67])
#Log of supporters at distance exactly 3 (different sites) divided by PageRank, mp

            log_siteneighbors_2_hp_siteneighbors_1_hp_pagerank_hp.append(row[68])
#Log of supporters at distance exactly 2 (different sites) divided by PageRank, hp

            log_siteneighbors_2_hp_siteneighbors_1_hp_pagerank_mp.append(row[69])
#Log of supporters at distance exactly 2 (different sites) divided by PageRank, mp

            siteneighbors_1_hp_divided_siteneighbors_1_mp.append(row[70])
#Supporters at distance 1 (different sites) of hp over mp

            siteneighbors_2_hp_divided_siteneighbors_2_mp.append(row[71])
#Supporters at distance 2 (different sites) of hp over mp

            siteneighbors_3_hp_divided_siteneighbors_3_mp.append(row[72])
#Supporters at distance 3 (different sites) of hp over mp

            siteneighbors_4_hp_divided_siteneighbors_4_mp.append(row[73])
#Supporters at distance 4 (different sites) of hp over mp

            log_neighbors_2_hp.append(row[74])
#Log of supporters at distance 2, hp (note: supporters at distance 1 is indegree)

            log_neighbors_3_hp.append(row[75])
#Log of supporters at distance 3, hp

            log_neighbors_4_hp.append(row[76])
#Log of supporters at distance 4, hp

            log_neighbors_2_mp.append(row[77])
#Log of supporters at distance 2, mp

            log_neighbors_3_mp.append(row[78])
#Log of supporters at distance 3, mp

            log_neighbors_4_mp.append(row[79])
#Log of supporters at distance 4, mp

            log_neighbors_2_hp_divided_pagerank_hp.append(row[80])
#Log of supporters at distance 2 divided by PageRank, hp

            log_neighbors_3_hp_divided_pagerank_hp.append(row[81])
#Log of supporters at distance 3 divided by PageRank, hp

            log_neighbors_4_hp_divided_pagerank_hp.append(row[82])
#Log of supporters at distance 4 divided by PageRank, hp

            log_neighbors_2_mp_divided_pagerank_mp.append(row[83])
#Log of supporters at distance 2 divided by PageRank, mp

            log_neighbors_3_mp_divided_pagerank_mp.append(row[84])
#Log of supporters at distance 3 divided by PageRank, mp

            log_neighbors_4_mp_divided_pagerank_mp.append(row[85])
#Log of supporters at distance 4 divided by PageRank, mp

            log_neighbors_4_hp_divided_neighbors_3_hp.append(row[86])
#Log of supporters at distance 4 divided by supporters at distance 3, hp

            log_neighbors_4_mp_divided_neighbors_3_mp.append(row[87])
#Log of supporters at distance 4 divided by supporters at distance 3, mp

            log_neighbors_3_hp_divided_neighbors_2_hp.append(row[88])
#Log of supporters at distance 3 divided by supporters at distance 2, hp

            log_neighbors_3_mp_divided_neighbors_2_mp.append(row[89])
#Log of supporters at distance 3 divided by supporters at distance 2, mp

            log_neighbors_2_hp_divided_indegree_hp.append(row[90])
#Log of supporters at distance 2 divided by supporters at distance 1, hp

            log_neighbors_2_mp_divided_indegree_mp.append(row[91])
#Log of supporters at distance 2 divided by supporters at distance 1, mp

            log_min_neighbors_hp.append(row[92])
#Log of minimum change of number of supporters at distance i over supporters at distance i-1, i=2..4, hp

            log_min_neighbors_mp.append(row[93])
#Log of minimum change of number of supporters at distance i over supporters at distance i-1, i=2..4, mp

            log_max_neighbors_hp.append(row[94])
#Log of maximum change of number of supporters at distance i over supporters at distance i-1, i=2..4, hp

            log_max_neighbors_mp.append(row[95])
#Log of maximum change of number of supporters at distance i over supporters at distance i-1, i=2..4, mp

            log_avg_neighbors_hp.append(row[96])
#Log of average change of number of supporters at distance i over supporters at distance i-1, i=2..4, hp

            log_avg_neighbors_mp.append(row[97])
#Log of average change of number of supporters at distance i over supporters at distance i-1, i=2..4, mp

            log_neighbors_4_divided_pagerank_hp.append(row[98])
#Log of number of supporters at distance exactly 4 over pagerank, hp

            log_neighbors_4_divided_pagerank_mp.append(row[99])
#Log of number of supporters at distance exactly 4 over pagerank, mp

            log_neighbors_3_divided_pagerank_hp.append(row[100])
#Log of number of supporters at distance exactly 3 over pagerank, hp

            log_neighbors_3_divided_pagerank_mp.append(row[101])
#Log of number of supporters at distance exactly 3 over pagerank, mp

            log_neighbors_2_divided_pagerank_hp.append(row[102])
#Log of number of supporters at distance exactly 2 over pagerank, hp

            log_neighbors_2_divided_pagerank_mp.append(row[103])
#Log of number of supporters at distance exactly 2 over pagerank, mp

            neighbors_2_hp_divided_neighbors_2_mp.append(row[104])
#Supporters at 2 in hp divided by supporters at 2.append(row[])

            neighbors_3_hp_divided_neighbors_3_mp.append(row[105])
#Supporters at 3 in hp divided by supporters at 4.append(row[])

            neighbors_4_hp_divided_neighbors_4_mp.append(row[106])
#Supporters at 3 in hp divided by supporters at 4.append(row[])

            log_truncatedpagerank_1_hp.append(row[107])
#Log of TruncatedPageRank with T=.append(row[])

            log_truncatedpagerank_2_hp.append(row[108])
#Log of TruncatedPageRank with T=2, hp

            log_truncatedpagerank_3_hp.append(row[109])
#Log of TruncatedPageRank with T=3, hp

            log_truncatedpagerank_4_hp.append(row[110])
#Log of TruncatedPageRank with T=4, hp

            log_truncatedpagerank_1_mp.append(row[111])
#Log of TruncatedPageRank with T=1, mp

            log_truncatedpagerank_2_mp.append(row[112])
#Log of TruncatedPageRank with T=2, mp

            log_truncatedpagerank_3_mp.append(row[113])
#Log of TruncatedPageRank with T=3, mp

            log_truncatedpagerank_4_mp.append(row[114])
#Log of TruncatedPageRank with T=4, mp

            log_truncatedpagerank_1_hp_divided_pagerank_hp.append(row[115])
#Log of TruncatedPageRank with T=1 divided by PageRank, hp

            log_truncatedpagerank_2_hp_divided_pagerank_hp.append(row[116])
#Log of TruncatedPageRank with T=2 divided by PageRank, hp

            log_truncatedpagerank_3_hp_divided_pagerank_hp.append(row[117])
#Log of TruncatedPageRank with T=3 divided by PageRank, hp

            log_truncatedpagerank_4_hp_divided_pagerank_hp.append(row[118])
#Log of TruncatedPageRank with T=4 divided by PageRank, hp

            log_truncatedpagerank_1_mp_divided_pagerank_mp.append(row[119])
#Log of TruncatedPageRank with T=1 divided by PageRank, mp

            log_truncatedpagerank_2_mp_divided_pagerank_mp.append(row[120])
#Log of TruncatedPageRank with T=2 divided by PageRank, mp

            log_truncatedpagerank_3_mp_divided_pagerank_mp.append(row[121])
#Log of TruncatedPageRank with T=3 divided by PageRank, mp

            log_truncatedpagerank_4_mp_divided_pagerank_mp.append(row[122])
#Log of TruncatedPageRank with T=4 divided by PageRank, mp

            truncatedpagerank_4_hp_divided_truncatedpagerank_3_hp.append(row[123])
#Log of TruncatedPageRank with T=4 divided by TruncatedPageRank with T=3, hp

            truncatedpagerank_4_mp_divided_truncatedpagerank_3_mp.append(row[124])
#Log of TruncatedPageRank with T=4 divided by TruncatedPageRank with T=3, mp

            truncatedpagerank_3_hp_divided_truncatedpagerank_2_hp.append(row[125])
#Log of TruncatedPageRank with T=3 divided by TruncatedPageRank with T=3, hp

            truncatedpagerank_3_mp_divided_truncatedpagerank_2_mp.append(row[126])
#Log of TruncatedPageRank with T=3 divided by TruncatedPageRank with T=3, mp

            truncatedpagerank_2_hp_divided_truncatedpagerank_1_hp.append(row[127])
#Log of TruncatedPageRank with T=2 divided by TruncatedPageRank with T=3, hp

            truncatedpagerank_2_mp_divided_truncatedpagerank_1_mp.append(row[128])
#Log of TruncatedPageRank with T=2 divided by TruncatedPageRank with T=3, mp

            log_min_truncatedpagerank_hp.append(row[129])
#Log of minimum of TruncatedPageRank with T=i over TruncatedPageRank with T=i-1, i=2..4, hp

            log_min_truncatedpagerank_mp.append(row[130])
#Log of minimum of TruncatedPageRank with T=i over TruncatedPageRank with T=i-1, i=2..4, mp

            log_max_truncatedpagerank_hp.append(row[131])
#Log of maximum of TruncatedPageRank with T=i over TruncatedPageRank with T=i-1, i=2..4, hp

            log_max_truncatedpagerank_mp.append(row[132])
#Log of maximum of TruncatedPageRank with T=i over TruncatedPageRank with T=i-1, i=2..4, mp

            log_avg_truncatedpagerank_hp.append(row[133])
#Log of average of TruncatedPageRank with T=i over TruncatedPageRank with T=i-1, i=2..4, hp

            log_avg_truncatedpagerank_mp.append(row[134])
#Log of average of TruncatedPageRank with T=i over TruncatedPageRank with T=i-1, i=2..4, mp

            truncatedpagerank_1_mp_divided_truncatedpagerank_1_hp.append(row[135])
#TruncatedPageRank with T=1 in hp, divided by TruncatedPageRank with T=1 in mp

            truncatedpagerank_2_mp_divided_truncatedpagerank_2_hp.append(row[136])
#TruncatedPageRank with T=2 in hp, divided by TruncatedPageRank with T=2 in mp

            truncatedpagerank_3_mp_divided_truncatedpagerank_3_hp.append(row[137])
#TruncatedPageRank with T=3 in hp, divided by TruncatedPageRank with T=3 in mp

            truncatedpagerank_4_mp_divided_truncatedpagerank_4_hp.append(row[138])
#TruncatedPageRank with T=4 in hp, divided by TruncatedPageRank with T=4 in mp
 
        # Increase the row number
        rowNr = rowNr + 1
        
f.close()


# In[698]:

######### train other graph-based features #########
t_log_indegree_hp = []

t_log_indegree_mp = []

t_log_outdegree_hp = []

t_log_outdegree_mp = []

t_reciprocity_hp = []

t_reciprocity_mp = []

t_log_assortativity_hp = []

t_log_assortativity_mp = []

t_log_avgin_of_out_hp = []

t_log_avgin_of_out_mp = []

t_log_avgin_of_out_hp_times_outdegree_hp = []

t_log_avgin_of_out_hp_times_outdegree_mp = []

t_log_avgout_of_in_hp = []

t_log_avgout_of_in_mp = []

t_log_avgout_of_in_hp_times_indegree_hp = []

t_log_avgout_of_in_hp_times_indegree_mp = []

t_eq_hp_mp = []

t_log_pagerank_hp = []

t_log_pagerank_mp = []

t_log_indegree_hp_divided_pagerank_hp = []

t_log_indegree_mp_divided_pagerank_mp = []

t_log_outdegree_hp_divided_pagerank_hp = []

t_log_outdegree_mp_divided_pagerank_mp = []

t_log_prsigma_hp = []

t_log_prsigma_mp = []

t_log_prsigma_hp_divided_pagerank_hp = []

t_log_prsigma_mp_divided_pagerank_mp = []

t_pagerank_mp_divided_pagerank_hp = []

t_log_trustrank_hp = []

t_log_trustrank_mp = []

t_log_trustrank_hp_divided_pagerank_hp = []

t_log_trustrank_mp_divided_pagerank_mp = []

t_log_trustrank_hp_divided_indegree_hp = []

t_log_trustrank_mp_divided_indegree_mp = []

t_trustrank_mp_divided_trustrank_hp = []

t_log_siteneighbors_1_hp = []

t_log_siteneighbors_2_hp = []

t_log_siteneighbors_3_hp = []

t_log_siteneighbors_4_hp = []

t_log_siteneighbors_1_mp = []

t_log_siteneighbors_2_mp = []

t_log_siteneighbors_3_mp = []

t_log_siteneighbors_4_mp = []

t_log_siteneighbors_1_hp_divided_pagerank_hp = []

t_log_siteneighbors_2_hp_divided_pagerank_hp = []

t_log_siteneighbors_3_hp_divided_pagerank_hp = []

t_log_siteneighbors_4_hp_divided_pagerank_hp = []

t_log_siteneighbors_1_mp_divided_pagerank_mp = []

t_log_siteneighbors_2_mp_divided_pagerank_mp = []

t_log_siteneighbors_3_mp_divided_pagerank_mp = []

t_log_siteneighbors_4_mp_divided_pagerank_mp = []

t_log_siteneighbors_4_hp_divided_siteneighbors_3_hp = []

t_log_siteneighbors_4_mp_divided_siteneighbors_3_mp = []

t_log_siteneighbors_3_hp_divided_siteneighbors_2_hp = []

t_log_siteneighbors_3_mp_divided_siteneighbors_2_mp = []

t_log_siteneighbors_2_hp_divided_siteneighbors_1_hp = []

t_log_siteneighbors_2_mp_divided_siteneighbors_1_mp = []

t_log_min_siteneighbors_hp = []

t_log_min_siteneighbors_mp = []

t_log_max_siteneighbors_hp = []

t_log_max_siteneighbors_mp = []

t_log_avg_siteneighbors_hp = []

t_log_avg_siteneighbors_mp = []

t_log_siteneighbors_4_hp_siteneighbors_3_hp_pagerank_hp = []

t_log_siteneighbors_4_hp_siteneighbors_3_mp_pagerank_mp = []

t_log_siteneighbors_3_hp_siteneighbors_2_hp_pagerank_hp = []

t_log_siteneighbors_3_hp_siteneighbors_2_mp_pagerank_mp = []

t_log_siteneighbors_2_hp_siteneighbors_1_hp_pagerank_hp = []

t_log_siteneighbors_2_hp_siteneighbors_1_hp_pagerank_mp = []

t_siteneighbors_1_hp_divided_siteneighbors_1_mp = []

t_siteneighbors_2_hp_divided_siteneighbors_2_mp = []

t_siteneighbors_3_hp_divided_siteneighbors_3_mp = []

t_siteneighbors_4_hp_divided_siteneighbors_4_mp = []

t_log_neighbors_2_hp = []

t_log_neighbors_3_hp = []

t_log_neighbors_4_hp = []

t_log_neighbors_2_mp = []

t_log_neighbors_3_mp = []

t_log_neighbors_4_mp = []

t_log_neighbors_2_hp_divided_pagerank_hp = []

t_log_neighbors_3_hp_divided_pagerank_hp = []

t_log_neighbors_4_hp_divided_pagerank_hp = []

t_log_neighbors_2_mp_divided_pagerank_mp = []

t_log_neighbors_3_mp_divided_pagerank_mp = []

t_log_neighbors_4_mp_divided_pagerank_mp = []

t_log_neighbors_4_hp_divided_neighbors_3_hp = []

t_log_neighbors_4_mp_divided_neighbors_3_mp = []

t_log_neighbors_3_hp_divided_neighbors_2_hp = []

t_log_neighbors_3_mp_divided_neighbors_2_mp = []

t_log_neighbors_2_hp_divided_indegree_hp = []

t_log_neighbors_2_mp_divided_indegree_mp = []

t_log_min_neighbors_hp = []

t_log_min_neighbors_mp = []

t_log_max_neighbors_hp = []

t_log_max_neighbors_mp = []

t_log_avg_neighbors_hp = []

t_log_avg_neighbors_mp = []

t_log_neighbors_4_divided_pagerank_hp = []

t_log_neighbors_4_divided_pagerank_mp = []

t_log_neighbors_3_divided_pagerank_hp = []

t_log_neighbors_3_divided_pagerank_mp = []

t_log_neighbors_2_divided_pagerank_hp = []

t_log_neighbors_2_divided_pagerank_mp = []

t_neighbors_2_hp_divided_neighbors_2_mp = []

t_neighbors_3_hp_divided_neighbors_3_mp = []

t_neighbors_4_hp_divided_neighbors_4_mp = []

t_log_truncatedpagerank_1_hp = []

t_log_truncatedpagerank_2_hp = []

t_log_truncatedpagerank_3_hp = []

t_log_truncatedpagerank_4_hp = []

t_log_truncatedpagerank_1_mp = []

t_log_truncatedpagerank_2_mp = []

t_log_truncatedpagerank_3_mp = []

t_log_truncatedpagerank_4_mp = []

t_log_truncatedpagerank_1_hp_divided_pagerank_hp = []

t_log_truncatedpagerank_2_hp_divided_pagerank_hp = []

t_log_truncatedpagerank_3_hp_divided_pagerank_hp = []

t_log_truncatedpagerank_4_hp_divided_pagerank_hp = []

t_log_truncatedpagerank_1_mp_divided_pagerank_mp = []

t_log_truncatedpagerank_2_mp_divided_pagerank_mp = []

t_log_truncatedpagerank_3_mp_divided_pagerank_mp = []

t_log_truncatedpagerank_4_mp_divided_pagerank_mp = []

t_truncatedpagerank_4_hp_divided_truncatedpagerank_3_hp = []

t_truncatedpagerank_4_mp_divided_truncatedpagerank_3_mp = []

t_truncatedpagerank_3_hp_divided_truncatedpagerank_2_hp = []

t_truncatedpagerank_3_mp_divided_truncatedpagerank_2_mp = []

t_truncatedpagerank_2_hp_divided_truncatedpagerank_1_hp = []

t_truncatedpagerank_2_mp_divided_truncatedpagerank_1_mp = []

t_log_min_truncatedpagerank_hp = []

t_log_min_truncatedpagerank_mp = []

t_log_max_truncatedpagerank_hp = []

t_log_max_truncatedpagerank_mp = []

t_log_avg_truncatedpagerank_hp = []

t_log_avg_truncatedpagerank_mp = []

t_truncatedpagerank_1_mp_divided_truncatedpagerank_1_hp = []

t_truncatedpagerank_2_mp_divided_truncatedpagerank_2_hp = []

t_truncatedpagerank_3_mp_divided_truncatedpagerank_3_hp = []

t_truncatedpagerank_4_mp_divided_truncatedpagerank_4_hp = []


# In[699]:

for ids in id_label:
    
    t_log_indegree_hp.append(log_indegree_hp[int(ids)])
    
    t_log_indegree_mp.append(log_indegree_mp[int(ids)])
    
    t_log_outdegree_hp.append(log_outdegree_hp[int(ids)])
    
    t_log_outdegree_mp.append(log_outdegree_mp[int(ids)])
    
    t_reciprocity_hp.append(reciprocity_hp[int(ids)])
    
    t_reciprocity_mp.append(reciprocity_mp[int(ids)])
    
    t_log_assortativity_hp.append(log_assortativity_hp[int(ids)])
    
    t_log_assortativity_mp.append(log_assortativity_mp[int(ids)])
    
    t_log_avgin_of_out_hp.append(log_avgin_of_out_hp[int(ids)])
    
    t_log_avgin_of_out_mp.append(log_avgin_of_out_mp[int(ids)]) 
    
    t_log_avgin_of_out_hp_times_outdegree_hp.append(log_avgin_of_out_hp_times_outdegree_hp[int(ids)]) 
    
    t_log_avgin_of_out_hp_times_outdegree_mp.append(log_avgin_of_out_hp_times_outdegree_mp[int(ids)]) 
    
    t_log_avgout_of_in_hp.append(log_avgout_of_in_hp[int(ids)]) 
    
    t_log_avgout_of_in_mp.append(log_avgout_of_in_mp[int(ids)]) 
    
    t_log_avgout_of_in_hp_times_indegree_hp.append(log_avgout_of_in_hp_times_indegree_hp[int(ids)])
    
    t_log_avgout_of_in_hp_times_indegree_mp.append(log_avgout_of_in_hp_times_indegree_mp[int(ids)])
    
    t_eq_hp_mp.append(eq_hp_mp[int(ids)])
    
    t_log_pagerank_hp.append(log_pagerank_hp[int(ids)])
    
    t_log_pagerank_mp.append(log_pagerank_mp[int(ids)])
    
    t_log_indegree_hp_divided_pagerank_hp.append(log_indegree_hp_divided_pagerank_hp[int(ids)])
    
    t_log_indegree_mp_divided_pagerank_mp.append(log_indegree_mp_divided_pagerank_mp[int(ids)])
    
    t_log_outdegree_hp_divided_pagerank_hp.append(log_outdegree_hp_divided_pagerank_hp[int(ids)])
    
    t_log_outdegree_mp_divided_pagerank_mp.append(log_outdegree_mp_divided_pagerank_mp[int(ids)])
    
    t_log_prsigma_hp.append(log_prsigma_hp[int(ids)])
    
    t_log_prsigma_mp.append(log_prsigma_mp[int(ids)])
    
    t_log_prsigma_hp_divided_pagerank_hp.append(log_prsigma_hp_divided_pagerank_hp[int(ids)])
    
    t_log_prsigma_mp_divided_pagerank_mp.append(log_prsigma_mp_divided_pagerank_mp[int(ids)])
    
    t_pagerank_mp_divided_pagerank_hp.append(pagerank_mp_divided_pagerank_hp[int(ids)])
    
    t_log_trustrank_hp.append(log_trustrank_hp[int(ids)])
    
    t_log_trustrank_mp.append(log_trustrank_mp[int(ids)])
    
    t_log_trustrank_hp_divided_pagerank_hp.append(log_trustrank_hp_divided_pagerank_hp[int(ids)])
    
    t_log_trustrank_mp_divided_pagerank_mp.append(log_trustrank_mp_divided_pagerank_mp[int(ids)])
    
    t_log_trustrank_hp_divided_indegree_hp.append(log_trustrank_hp_divided_indegree_hp[int(ids)])
    
    t_log_trustrank_mp_divided_indegree_mp.append(log_trustrank_mp_divided_indegree_mp[int(ids)])
    
    t_trustrank_mp_divided_trustrank_hp.append(trustrank_mp_divided_trustrank_hp[int(ids)])
    
    t_log_siteneighbors_1_hp.append(log_siteneighbors_1_hp[int(ids)])
    
    t_log_siteneighbors_2_hp.append(log_siteneighbors_2_hp[int(ids)])
    
    t_log_siteneighbors_3_hp.append(log_siteneighbors_3_hp[int(ids)])
    
    t_log_siteneighbors_4_hp.append(log_siteneighbors_4_hp[int(ids)])
    
    t_log_siteneighbors_1_mp.append(log_siteneighbors_1_mp[int(ids)])
    
    t_log_siteneighbors_2_mp.append(log_siteneighbors_2_mp[int(ids)])
    
    t_log_siteneighbors_3_mp.append(log_siteneighbors_3_mp[int(ids)])
    
    t_log_siteneighbors_4_mp.append(log_siteneighbors_4_mp[int(ids)])
    
    t_log_siteneighbors_1_hp_divided_pagerank_hp.append(log_siteneighbors_1_hp_divided_pagerank_hp[int(ids)])
    
    t_log_siteneighbors_2_hp_divided_pagerank_hp.append(log_siteneighbors_2_hp_divided_pagerank_hp[int(ids)])
    
    t_log_siteneighbors_3_hp_divided_pagerank_hp.append(log_siteneighbors_3_hp_divided_pagerank_hp[int(ids)])
    
    t_log_siteneighbors_4_hp_divided_pagerank_hp.append(log_siteneighbors_4_hp_divided_pagerank_hp[int(ids)])
    
    t_log_siteneighbors_1_mp_divided_pagerank_mp.append(log_siteneighbors_1_mp_divided_pagerank_mp[int(ids)])
    
    t_log_siteneighbors_2_mp_divided_pagerank_mp.append(log_siteneighbors_2_mp_divided_pagerank_mp[int(ids)])
    
    t_log_siteneighbors_3_mp_divided_pagerank_mp.append(log_siteneighbors_3_mp_divided_pagerank_mp[int(ids)])
    
    t_log_siteneighbors_4_mp_divided_pagerank_mp.append(log_siteneighbors_4_mp_divided_pagerank_mp[int(ids)])
    
    t_log_siteneighbors_4_hp_divided_siteneighbors_3_hp.append(log_siteneighbors_4_hp_divided_siteneighbors_3_hp[int(ids)])
    
    t_log_siteneighbors_4_mp_divided_siteneighbors_3_mp.append(log_siteneighbors_4_mp_divided_siteneighbors_3_mp[int(ids)])
    
    t_log_siteneighbors_3_hp_divided_siteneighbors_2_hp.append(log_siteneighbors_3_hp_divided_siteneighbors_2_hp[int(ids)])
    
    t_log_siteneighbors_3_mp_divided_siteneighbors_2_mp.append(log_siteneighbors_3_mp_divided_siteneighbors_2_mp[int(ids)])
    
    t_log_siteneighbors_2_hp_divided_siteneighbors_1_hp.append(log_siteneighbors_2_hp_divided_siteneighbors_1_hp[int(ids)])
    
    t_log_siteneighbors_2_mp_divided_siteneighbors_1_mp.append(log_siteneighbors_2_mp_divided_siteneighbors_1_mp[int(ids)])
    
    t_log_min_siteneighbors_hp.append(log_min_siteneighbors_hp[int(ids)])
    
    t_log_min_siteneighbors_mp.append(log_min_siteneighbors_mp[int(ids)])
    
    t_log_max_siteneighbors_hp.append(log_max_siteneighbors_hp[int(ids)])
    
    t_log_max_siteneighbors_mp.append(log_max_siteneighbors_mp[int(ids)])
    
    t_log_avg_siteneighbors_hp.append(log_avg_siteneighbors_hp[int(ids)])
    
    t_log_avg_siteneighbors_mp.append(log_avg_siteneighbors_mp[int(ids)])
    
    t_log_siteneighbors_4_hp_siteneighbors_3_hp_pagerank_hp.append(log_siteneighbors_4_hp_siteneighbors_3_hp_pagerank_hp[int(ids)])
    
    t_log_siteneighbors_4_hp_siteneighbors_3_mp_pagerank_mp.append(log_siteneighbors_4_hp_siteneighbors_3_mp_pagerank_mp[int(ids)])
    
    t_log_siteneighbors_3_hp_siteneighbors_2_hp_pagerank_hp.append(log_siteneighbors_3_hp_siteneighbors_2_hp_pagerank_hp[int(ids)])
    
    t_log_siteneighbors_3_hp_siteneighbors_2_mp_pagerank_mp.append(log_siteneighbors_3_hp_siteneighbors_2_mp_pagerank_mp[int(ids)])
    
    t_log_siteneighbors_2_hp_siteneighbors_1_hp_pagerank_hp.append(log_siteneighbors_2_hp_siteneighbors_1_hp_pagerank_hp[int(ids)])
    
    t_log_siteneighbors_2_hp_siteneighbors_1_hp_pagerank_mp.append(log_siteneighbors_2_hp_siteneighbors_1_hp_pagerank_mp[int(ids)])
    
    t_siteneighbors_1_hp_divided_siteneighbors_1_mp.append(siteneighbors_1_hp_divided_siteneighbors_1_mp[int(ids)])
    #start at 71
    t_siteneighbors_2_hp_divided_siteneighbors_2_mp.append(siteneighbors_2_hp_divided_siteneighbors_2_mp[int(ids)])
    
    t_siteneighbors_3_hp_divided_siteneighbors_3_mp.append(siteneighbors_3_hp_divided_siteneighbors_3_mp[int(ids)])

    t_siteneighbors_4_hp_divided_siteneighbors_4_mp.append(siteneighbors_4_hp_divided_siteneighbors_4_mp[int(ids)])

    t_log_neighbors_2_hp.append(log_neighbors_2_hp[int(ids)])

    t_log_neighbors_3_hp.append(log_neighbors_3_hp[int(ids)])

    t_log_neighbors_4_hp.append(log_neighbors_4_hp[int(ids)])

    t_log_neighbors_2_mp.append(log_neighbors_2_mp[int(ids)])

    t_log_neighbors_3_mp.append(log_neighbors_3_mp[int(ids)])

    t_log_neighbors_4_mp.append(log_neighbors_4_mp[int(ids)])

    t_log_neighbors_2_hp_divided_pagerank_hp.append(log_neighbors_2_hp_divided_pagerank_hp[int(ids)])

    t_log_neighbors_3_hp_divided_pagerank_hp.append(log_neighbors_3_hp_divided_pagerank_hp[int(ids)])

    t_log_neighbors_4_hp_divided_pagerank_hp.append(log_neighbors_4_hp_divided_pagerank_hp[int(ids)])

    t_log_neighbors_2_mp_divided_pagerank_mp.append(log_neighbors_2_mp_divided_pagerank_mp[int(ids)])

    t_log_neighbors_3_mp_divided_pagerank_mp.append(log_neighbors_3_mp_divided_pagerank_mp[int(ids)])

    t_log_neighbors_4_mp_divided_pagerank_mp.append(log_neighbors_4_mp_divided_pagerank_mp[int(ids)])

    t_log_neighbors_4_hp_divided_neighbors_3_hp.append(log_neighbors_4_hp_divided_neighbors_3_hp[int(ids)])

    t_log_neighbors_4_mp_divided_neighbors_3_mp.append(log_neighbors_4_mp_divided_neighbors_3_mp[int(ids)])

    t_log_neighbors_3_hp_divided_neighbors_2_hp.append(log_neighbors_3_hp_divided_neighbors_2_hp[int(ids)])

    t_log_neighbors_3_mp_divided_neighbors_2_mp.append(log_neighbors_3_mp_divided_neighbors_2_mp[int(ids)])

    t_log_neighbors_2_hp_divided_indegree_hp.append(log_neighbors_2_hp_divided_indegree_hp[int(ids)])

    t_log_neighbors_2_mp_divided_indegree_mp.append(log_neighbors_2_mp_divided_indegree_mp[int(ids)])

    t_log_min_neighbors_hp.append(log_min_neighbors_hp[int(ids)])

    t_log_min_neighbors_mp.append(log_min_neighbors_mp[int(ids)])

    t_log_max_neighbors_hp.append(log_max_neighbors_hp[int(ids)])

    t_log_max_neighbors_mp.append(log_max_neighbors_mp[int(ids)])
    
    t_log_avg_neighbors_hp.append(log_avg_neighbors_hp[int(ids)])
    
    t_log_avg_neighbors_mp.append(log_avg_neighbors_mp[int(ids)])

    t_log_neighbors_4_divided_pagerank_hp.append(log_neighbors_4_divided_pagerank_hp[int(ids)])

    t_log_neighbors_4_divided_pagerank_mp.append(log_neighbors_4_divided_pagerank_mp[int(ids)])

    t_log_neighbors_3_divided_pagerank_hp.append(log_neighbors_3_divided_pagerank_hp[int(ids)])

    t_log_neighbors_3_divided_pagerank_mp.append(log_neighbors_3_divided_pagerank_mp[int(ids)])

    t_log_neighbors_2_divided_pagerank_hp.append(log_neighbors_2_divided_pagerank_hp[int(ids)])

    t_log_neighbors_2_divided_pagerank_mp.append(log_neighbors_2_divided_pagerank_mp[int(ids)])

    t_neighbors_2_hp_divided_neighbors_2_mp.append(neighbors_2_hp_divided_neighbors_2_mp[int(ids)])

    t_neighbors_3_hp_divided_neighbors_3_mp.append(neighbors_3_hp_divided_neighbors_3_mp[int(ids)])

    t_neighbors_4_hp_divided_neighbors_4_mp.append(neighbors_4_hp_divided_neighbors_4_mp[int(ids)])

    t_log_truncatedpagerank_1_hp.append(log_truncatedpagerank_1_hp[int(ids)])

    t_log_truncatedpagerank_2_hp.append(log_truncatedpagerank_2_hp[int(ids)])

    t_log_truncatedpagerank_3_hp.append(log_truncatedpagerank_3_hp[int(ids)])

    t_log_truncatedpagerank_4_hp.append(log_truncatedpagerank_4_hp[int(ids)])

    t_log_truncatedpagerank_1_mp.append(log_truncatedpagerank_1_mp[int(ids)])

    t_log_truncatedpagerank_2_mp.append(log_truncatedpagerank_2_mp[int(ids)])

    t_log_truncatedpagerank_3_mp.append(log_truncatedpagerank_3_mp[int(ids)])

    t_log_truncatedpagerank_4_mp.append(log_truncatedpagerank_4_mp[int(ids)])

    t_log_truncatedpagerank_1_hp_divided_pagerank_hp.append(log_truncatedpagerank_1_hp_divided_pagerank_hp[int(ids)])

    t_log_truncatedpagerank_2_hp_divided_pagerank_hp.append(log_truncatedpagerank_2_hp_divided_pagerank_hp[int(ids)])

    t_log_truncatedpagerank_3_hp_divided_pagerank_hp.append(log_truncatedpagerank_3_hp_divided_pagerank_hp[int(ids)])

    t_log_truncatedpagerank_4_hp_divided_pagerank_hp.append(log_truncatedpagerank_4_hp_divided_pagerank_hp[int(ids)])

    t_log_truncatedpagerank_1_mp_divided_pagerank_mp.append(log_truncatedpagerank_1_mp_divided_pagerank_mp[int(ids)])

    t_log_truncatedpagerank_2_mp_divided_pagerank_mp.append(log_truncatedpagerank_2_mp_divided_pagerank_mp[int(ids)])

    t_log_truncatedpagerank_3_mp_divided_pagerank_mp.append(log_truncatedpagerank_3_mp_divided_pagerank_mp[int(ids)])

    t_log_truncatedpagerank_4_mp_divided_pagerank_mp.append(log_truncatedpagerank_4_mp_divided_pagerank_mp[int(ids)])

    t_truncatedpagerank_4_hp_divided_truncatedpagerank_3_hp.append(truncatedpagerank_4_hp_divided_truncatedpagerank_3_hp[int(ids)])

    t_truncatedpagerank_4_mp_divided_truncatedpagerank_3_mp.append(truncatedpagerank_4_mp_divided_truncatedpagerank_3_mp[int(ids)])

    t_truncatedpagerank_3_hp_divided_truncatedpagerank_2_hp.append(truncatedpagerank_3_hp_divided_truncatedpagerank_2_hp[int(ids)])

    t_truncatedpagerank_3_mp_divided_truncatedpagerank_2_mp.append(truncatedpagerank_3_mp_divided_truncatedpagerank_2_mp[int(ids)])

    t_truncatedpagerank_2_hp_divided_truncatedpagerank_1_hp.append(truncatedpagerank_2_hp_divided_truncatedpagerank_1_hp[int(ids)])

    t_truncatedpagerank_2_mp_divided_truncatedpagerank_1_mp.append(truncatedpagerank_2_mp_divided_truncatedpagerank_1_mp[int(ids)])

    t_log_min_truncatedpagerank_hp.append(log_min_truncatedpagerank_hp[int(ids)])

    t_log_min_truncatedpagerank_mp.append(log_min_truncatedpagerank_mp[int(ids)])

    t_log_max_truncatedpagerank_hp.append(log_max_truncatedpagerank_hp[int(ids)])

    t_log_max_truncatedpagerank_mp.append(log_max_truncatedpagerank_mp[int(ids)])

    t_log_avg_truncatedpagerank_hp.append(log_avg_truncatedpagerank_hp[int(ids)])

    t_log_avg_truncatedpagerank_mp.append(log_avg_truncatedpagerank_mp[int(ids)])

    t_truncatedpagerank_1_mp_divided_truncatedpagerank_1_hp.append(truncatedpagerank_1_mp_divided_truncatedpagerank_1_hp[int(ids)])

    t_truncatedpagerank_2_mp_divided_truncatedpagerank_2_hp.append(truncatedpagerank_2_mp_divided_truncatedpagerank_2_hp[int(ids)])

    t_truncatedpagerank_3_mp_divided_truncatedpagerank_3_hp.append(truncatedpagerank_3_mp_divided_truncatedpagerank_3_hp[int(ids)])
    
    t_truncatedpagerank_4_mp_divided_truncatedpagerank_4_hp.append(truncatedpagerank_4_mp_divided_truncatedpagerank_4_hp[int(ids)])


# In[808]:

##################### content-based features ####################
HST_1 = []
#Number of words in the page (home page = hp)

HST_2 = []
#Number of words in the title (hp)

HST_3 = []
#Average word length (hp)

HST_4 = []
#Fraction of anchor text (hp)

HST_5 = []
#Fraction of visible text (hp)

HST_6 = []
#Compression rate of the hp

HST_7 = []
#Top 100 corpus precision (hp)

HST_8 = []
#Top 200 corpus precision (hp)

HST_9 = []
#Top 500 corpus precision (hp)

HST_10 = []
#Top 1000 corpus precision (hp)

HST_11 = []
#Top 100 corpus recall (hp)

HST_12 = []
#Top 200 corpus recall (hp)

HST_13 = []
#Top 500 corpus recall (hp)

HST_14 = []
#Top 1000 corpus recall (hp)

HST_15 = []
#Top 100 queries precision (hp)

HST_16 = []
#Top 200 queries precision (hp)

HST_17 = []
#Top 500 queries precision (hp)

HST_18 = []
#Top 1000 queries precision (hp)

HST_19 = []
#Top 100 queries recall (hp)

HST_20 = []
#Top 200 queries recall (hp)

HST_21 = []
#Top 500 queries recall (hp)

HST_22 = []
#Top 1000 queries recall (hp)

HST_23 = []
#Entropy (hp)

HST_24 = []
#Independent LH (hp)

HMG_25 = []
#Number of words in the page (page with max PageRank in the host = mp)

HMG_26 = []
#Number of words in the title (mp)

HMG_27 = []
#Average word length (mp)

HMG_28 = []
#Fraction of anchor text (mp)

HMG_29 = []
#Fraction of visible text (mp)

HMG_30 = []
#Compression rate (mp)

HMG_31 = []
#Top 100 corpus precision (mp)

HMG_32 = []
#Top 200 corpus precision (mp)

HMG_33 = []
#Top 500 corpus precision (mp)

HMG_34 = []
#Top 1000 corpus precision (mp)

HMG_35 = []
#Top 100 corpus recall (mp)

HMG_36 = []
#Top 200 corpus recall (mp)

HMG_37 = []
#Top 500 corpus recall (mp)

HMG_38 = []
#Top 1000 corpus recall (mp)

HMG_39 = []
#Top 100 queries precision (mp)

HMG_40 = []
#Top 200 queries precision (mp)

HMG_41 = []
#Top 500 queries precision (mp)

HMG_42 = []
#Top 1000 queries precision (mp)

HMG_43 = []
#Top 100 queries recall (mp)

HMG_44 = []
#Top 200 queries recall (mp)

HMG_45 = []
#Top 500 queries recall (mp)

HMG_46 = []
#Top 1000 queries recall (mp)

HMG_47 = []
#Entropy (mp)

HMG_48 = []
#Independent LH (mp)

AVG_49 = []
#Number of words in the page (average value for all pages in the host)

AVG_50 = []
#Number of words in the title (average value for all pages in the host)

AVG_51 = []
#Average word length (average value for all pages in the host)

AVG_52 = []
#Fraction of anchor text (average value for all pages in the host)

AVG_53 = []
#Fraction of visible text (average value for all pages in the host)

AVG_54 = []
#Compression rate (average value for all pages in the host)

AVG_55 = []
#Top 100 corpus precision (average value for all pages in the host)

AVG_56 = []
#Top 200 corpus precision (average value for all pages in the host)

AVG_57 = []
#Top 500 corpus precision (average value for all pages in the host)

AVG_58 = []
#Top 1000 corpus precision (average value for all pages in the host)

AVG_59 = []
#Top 100 corpus recall (average value for all pages in the host)

AVG_60 = []
#Top 200 corpus recall (average value for all pages in the host)

AVG_61 = []
#Top 500 corpus recall (average value for all pages in the host)

AVG_62 = []
#Top 1000 corpus recall (average value for all pages in the host)

AVG_63 = []
#Top 100 queries precision (average value for all pages in the host)

AVG_64 = []
#Top 200 queries precision (average value for all pages in the host)

AVG_65 = []
#Top 500 queries precision (average value for all pages in the host)

AVG_66 = []
#Top 1000 queries precision (average value for all pages in the host)

AVG_67 = []
#Top 100 queries recall (average value for all pages in the host)

AVG_68 = []
#Top 200 queries recall (average value for all pages in the host)

AVG_69 = []
#Top 500 queries recall (average value for all pages in the host)

AVG_70 = []
#Top 1000 queries recall (average value for all pages in the host)

AVG_71 = []
#Entropy (average value for all pages in the host)

AVG_72 = []
#Independent LH (average value for all pages in the host)

STD_73 = []
#Number of words in the page (Standard deviation for all pages in the host)

STD_74 = []
#Number of words in the title (Standard deviation for all pages in the host)

STD_75 = []
#Average word length (Standard deviation for all pages in the host)

STD_76 = []
#Fraction of anchor text (Standard deviation for all pages in the host)

STD_77 = []
#Fraction of visible text (Standard deviation for all pages in the host)

STD_78 = []
#Compression rate in the home page (Standard deviation for all pages in the host)

STD_79 = []
#Top 100 corpus precision (Standard deviation for all pages in the host)

STD_80 = []
#Top 200 corpus precision (Standard deviation for all pages in the host)

STD_81 = []
#Top 500 corpus precision (Standard deviation for all pages in the host)

STD_82 = []
#Top 1000 corpus precision (Standard deviation for all pages in the host)

STD_83 = []
#Top 100 corpus recall (Standard deviation for all pages in the host)

STD_84 = []
#Top 200 corpus recall (Standard deviation for all pages in the host)

STD_85 = []
#Top 500 corpus recall (Standard deviation for all pages in the host)

STD_86 = []
#Top 1000 corpus recall (Standard deviation for all pages in the host)

STD_87 = []
#Top 100 queries precision (Standard deviation for all pages in the host)

STD_88 = []
#Top 200 queries precision (Standard deviation for all pages in the host)

STD_89 = []
#Top 500 queries precision (Standard deviation for all pages in the host)

STD_90 = []
#Top 1000 queries precision (Standard deviation for all pages in the host)

STD_91 = []
#Top 100 queries recall (Standard deviation for all pages in the host)

STD_92 = []
#Top 200 queries recall (Standard deviation for all pages in the host)

STD_93 = []
#Top 500 queries recall (Standard deviation for all pages in the host)

STD_94 = []
#Top 1000 queries recall (Standard deviation for all pages in the host)

STD_95 = []
#Entropy (Standard deviation for all pages in the host)

STD_96 = []
#Independent LH (Standard deviation for all pages in the host)

# open file

with open('uk-2007-05.content_based_features.csv', 'rU') as f:
    #reader = csv.reader(f)
    #reader = csv.reader(f, dialect=csv.excel_tab)
    reader = csv.reader(f, dialect='excel') 
 
    # read file row by row
    rowNr = 0
    for row in reader:
        HST_1.append(row[2])
        HST_2.append(row[3])
        HST_3.append(row[4])
        HST_4.append(row[5])
        HST_5.append(row[6])
        HST_6.append(row[7])
        HST_7.append(row[8])
        HST_8.append(row[9])
        HST_9.append(row[10])
        HST_10.append(row[11])
        HST_11.append(row[12])
        HST_12.append(row[13])
        HST_13.append(row[14])
        HST_14.append(row[15])
        HST_15.append(row[16])
        HST_16.append(row[17])
        HST_17.append(row[18])
        HST_18.append(row[19])
        HST_19.append(row[20])
        HST_20.append(row[21])
        HST_21.append(row[22])
        HST_22.append(row[23])
        HST_23.append(row[24])
        HST_24.append(row[25])
        HMG_25.append(row[26])
        HMG_26.append(row[27])
        HMG_27.append(row[28])
        HMG_28.append(row[29])
        HMG_29.append(row[30])
        HMG_30.append(row[31])
        HMG_31.append(row[32])
        HMG_32.append(row[33])
        HMG_33.append(row[34])
        HMG_34.append(row[35])
        HMG_35.append(row[36])
        HMG_36.append(row[37])
        HMG_37.append(row[38])
        HMG_38.append(row[39])
        HMG_39.append(row[40])
        HMG_40.append(row[41])
        HMG_41.append(row[42])
        HMG_42.append(row[43])
        HMG_43.append(row[44])
        HMG_44.append(row[45])
        HMG_45.append(row[46])
        HMG_46.append(row[47])
        HMG_47.append(row[48])
        HMG_48.append(row[49])
        AVG_49.append(row[50])
        AVG_50.append(row[51])
        AVG_51.append(row[52])
        AVG_52.append(row[53])
        AVG_53.append(row[54])
        AVG_54.append(row[55])
        AVG_55.append(row[56])
        AVG_56.append(row[57])
        AVG_57.append(row[58])
        AVG_58.append(row[59])
        AVG_59.append(row[60])
        AVG_60.append(row[61])
        AVG_61.append(row[62])
        AVG_62.append(row[63])
        AVG_63.append(row[64])
        AVG_64.append(row[65])
        AVG_65.append(row[66])
        AVG_66.append(row[67])
        AVG_67.append(row[68])
        AVG_68.append(row[69])
        AVG_69.append(row[70])
        AVG_70.append(row[71])
        AVG_71.append(row[72])
        AVG_72.append(row[73])
        STD_73.append(row[74])
        STD_74.append(row[75])
        STD_75.append(row[76])
        STD_76.append(row[77])
        STD_77.append(row[78])
        STD_78.append(row[79])
        STD_79.append(row[80])
        STD_80.append(row[81])
        STD_81.append(row[82])
        STD_82.append(row[83])
        STD_83.append(row[84])
        STD_84.append(row[85])
        STD_85.append(row[86])
        STD_86.append(row[87])
        STD_87.append(row[88])
        STD_88.append(row[89])
        STD_89.append(row[90])
        STD_90.append(row[91])
        STD_91.append(row[92])
        STD_92.append(row[93])
        STD_93.append(row[94])
        STD_94.append(row[95])
        STD_95.append(row[96])
        STD_96.append(row[97])

  
        
f.close()


# In[813]:

###################### train content-based features ####################
t_HST_1 = []

t_HST_2 = []

t_HST_3 = []

t_HST_4 = []

t_HST_5 = []

t_HST_6 = []

t_HST_7 = []

t_HST_8 = []

t_HST_9 = []

t_HST_10 = []

t_HST_11 = []

t_HST_12 = []

t_HST_13 = []

t_HST_14 = []

t_HST_15 = []

t_HST_16 = []

t_HST_17 = []

t_HST_18 = []

t_HST_19 = []

t_HST_20 = []

t_HST_21 = []

t_HST_22 = []

t_HST_23 = []

t_HST_24 = []

t_HMG_25 = []

t_HMG_26 = []

t_HMG_27 = []

t_HMG_28 = []

t_HMG_29 = []

t_HMG_30 = []

t_HMG_31 = []

t_HMG_32 = []

t_HMG_33 = []

t_HMG_34 = []

t_HMG_35 = []

t_HMG_36 = []

t_HMG_37 = []

t_HMG_38 = []

t_HMG_39 = []

t_HMG_40 = []

t_HMG_41 = []

t_HMG_42 = []

t_HMG_43 = []

t_HMG_44 = []

t_HMG_45 = []

t_HMG_46 = []

t_HMG_47 = []

t_HMG_48 = []

t_AVG_49 = []

t_AVG_50 = []

t_AVG_51 = []

t_AVG_52 = []

t_AVG_53 = []

t_AVG_54 = []

t_AVG_55 = []

t_AVG_56 = []

t_AVG_57 = []

t_AVG_58 = []

t_AVG_59 = []

t_AVG_60 = []

t_AVG_61 = []

t_AVG_62 = []

t_AVG_63 = []

t_AVG_64 = []

t_AVG_65 = []

t_AVG_66 = []

t_AVG_67 = []

t_AVG_68 = []

t_AVG_69 = []

t_AVG_70 = []

t_AVG_71 = []

t_AVG_72 = []

t_STD_73 = []

t_STD_74 = []

t_STD_75 = []

t_STD_76 = []

t_STD_77 = []

t_STD_78 = []

t_STD_79 = []

t_STD_80 = []

t_STD_81 = []

t_STD_82 = []

t_STD_83 = []

t_STD_84 = []

t_STD_85 = []

t_STD_86 = []

t_STD_87 = []

t_STD_88 = []

t_STD_89 = []

t_STD_90 = []

t_STD_91 = []

t_STD_92 = []

t_STD_93 = []

t_STD_94 = []

t_STD_95 = []

t_STD_96 = []


# In[815]:

k=0
for ids in id_label:
    k=host_id.index(ids)
    t_HST_1.append(HST_1[k])
    t_HST_2.append(HST_2[k])
    t_HST_3.append(HST_3[k])
    t_HST_4.append(HST_4[k])
    t_HST_5.append(HST_5[k])
    t_HST_6.append(HST_6[k])
    t_HST_7.append(HST_7[k])
    t_HST_8.append(HST_8[k])
    t_HST_9.append(HST_9[k])
    t_HST_10.append(HST_10[k])
    t_HST_11.append(HST_11[k])
    t_HST_12.append(HST_12[k])
    t_HST_13.append(HST_13[k])
    t_HST_14.append(HST_14[k])
    t_HST_15.append(HST_15[k])
    t_HST_16.append(HST_16[k])
    t_HST_17.append(HST_17[k])
    t_HST_18.append(HST_18[k])
    t_HST_19.append(HST_19[k])
    t_HST_20.append(HST_20[k])
    t_HST_21.append(HST_21[k])
    t_HST_22.append(HST_22[k])
    t_HST_23.append(HST_23[k])
    t_HST_24.append(HST_24[k])
    t_HMG_25.append(HMG_25[k])
    t_HMG_26.append(HMG_26[k])
    t_HMG_27.append(HMG_27[k])
    t_HMG_28.append(HMG_28[k])
    t_HMG_29.append(HMG_29[k])
    t_HMG_30.append(HMG_30[k])
    t_HMG_31.append(HMG_31[k])
    t_HMG_32.append(HMG_32[k])
    t_HMG_33.append(HMG_33[k])
    t_HMG_34.append(HMG_34[k])
    t_HMG_35.append(HMG_35[k])
    t_HMG_36.append(HMG_36[k])
    t_HMG_37.append(HMG_37[k])
    t_HMG_38.append(HMG_38[k])
    t_HMG_39.append(HMG_39[k])
    t_HMG_40.append(HMG_40[k])
    t_HMG_41.append(HMG_41[k])
    t_HMG_42.append(HMG_42[k])
    t_HMG_43.append(HMG_43[k])
    t_HMG_44.append(HMG_44[k])
    t_HMG_45.append(HMG_45[k])
    t_HMG_46.append(HMG_46[k])
    t_HMG_47.append(HMG_47[k])
    t_HMG_48.append(HMG_48[k])
    t_AVG_49.append(AVG_49[k])
    t_AVG_50.append(AVG_50[k])
    t_AVG_51.append(AVG_51[k])
    t_AVG_52.append(AVG_52[k])
    t_AVG_53.append(AVG_53[k])
    t_AVG_54.append(AVG_54[k])
    t_AVG_55.append(AVG_55[k])
    t_AVG_56.append(AVG_56[k])
    t_AVG_57.append(AVG_57[k])
    t_AVG_58.append(AVG_58[k])
    t_AVG_59.append(AVG_59[k])
    t_AVG_60.append(AVG_60[k])
    t_AVG_61.append(AVG_61[k])
    t_AVG_62.append(AVG_62[k])
    t_AVG_63.append(AVG_63[k])
    t_AVG_64.append(AVG_64[k])
    t_AVG_65.append(AVG_65[k])
    t_AVG_66.append(AVG_66[k])
    t_AVG_67.append(AVG_67[k])
    t_AVG_68.append(AVG_68[k])
    t_AVG_69.append(AVG_69[k])
    t_AVG_70.append(AVG_70[k])
    t_AVG_71.append(AVG_71[k])
    t_AVG_72.append(AVG_72[k])
    t_STD_73.append(STD_73[k])
    t_STD_74.append(STD_74[k])
    t_STD_75.append(STD_75[k])
    t_STD_76.append(STD_76[k])
    t_STD_77.append(STD_77[k])
    t_STD_78.append(STD_78[k])
    t_STD_79.append(STD_79[k])
    t_STD_80.append(STD_80[k])
    t_STD_81.append(STD_81[k])
    t_STD_82.append(STD_82[k])
    t_STD_83.append(STD_83[k])
    t_STD_84.append(STD_84[k])
    t_STD_85.append(STD_85[k])
    t_STD_86.append(STD_86[k])
    t_STD_87.append(STD_87[k])
    t_STD_88.append(STD_88[k])
    t_STD_89.append(STD_89[k])
    t_STD_90.append(STD_90[k])
    t_STD_91.append(STD_91[k])
    t_STD_92.append(STD_92[k])
    t_STD_93.append(STD_93[k])
    t_STD_94.append(STD_94[k])
    t_STD_95.append(STD_95[k])
    t_STD_96.append(STD_96[k])
    


# In[900]:

#convert training features into array based on feature selection
training_features = np.array([t_log_indegree_hp,
                              t_log_indegree_mp,
                              t_log_outdegree_hp,
                              t_log_outdegree_mp,
                              #4t_reciprocity_hp,   
                              t_reciprocity_mp,
                              t_log_assortativity_hp,
                              t_log_assortativity_mp,
                              t_log_avgin_of_out_hp,
                              t_log_avgin_of_out_mp,
                              t_log_avgin_of_out_hp_times_outdegree_hp,
                              t_log_avgin_of_out_hp_times_outdegree_mp,
                              t_log_avgout_of_in_hp,
                              t_log_avgout_of_in_mp,
                              t_log_avgout_of_in_hp_times_indegree_hp,
                              t_log_avgout_of_in_hp_times_indegree_mp,
                              #16t_eq_hp_mp,
                              t_log_pagerank_hp,
                              t_log_pagerank_mp,
                              t_log_indegree_hp_divided_pagerank_hp,
                              t_log_indegree_mp_divided_pagerank_mp,
                              t_log_outdegree_hp_divided_pagerank_hp,
                              t_log_outdegree_mp_divided_pagerank_mp,
                              t_log_prsigma_hp,
                              t_log_prsigma_mp,
                              t_log_prsigma_hp_divided_pagerank_hp,
                              t_log_prsigma_mp_divided_pagerank_mp,
                              #27t_pagerank_mp_divided_pagerank_hp,
                              t_log_trustrank_hp,
                              t_log_trustrank_mp,
                              t_log_trustrank_hp_divided_pagerank_hp,
                              t_log_trustrank_mp_divided_pagerank_mp,
                              t_log_trustrank_hp_divided_indegree_hp,
                              t_log_trustrank_mp_divided_indegree_mp,
                              #34t_trustrank_mp_divided_trustrank_hp,
                              t_log_siteneighbors_1_hp,
                              t_log_siteneighbors_2_hp,
                              t_log_siteneighbors_3_hp,
                              t_log_siteneighbors_4_hp,
                              t_log_siteneighbors_1_mp,
                              t_log_siteneighbors_2_mp,
                              t_log_siteneighbors_3_mp,
                              t_log_siteneighbors_4_mp,
                              t_log_siteneighbors_1_hp_divided_pagerank_hp,
                              t_log_siteneighbors_2_hp_divided_pagerank_hp,
                              t_log_siteneighbors_3_hp_divided_pagerank_hp,
                              t_log_siteneighbors_4_hp_divided_pagerank_hp,
                              t_log_siteneighbors_1_mp_divided_pagerank_mp,
                              t_log_siteneighbors_2_mp_divided_pagerank_mp,
                              t_log_siteneighbors_3_mp_divided_pagerank_mp,
                              t_log_siteneighbors_4_mp_divided_pagerank_mp,
                              t_log_siteneighbors_4_hp_divided_siteneighbors_3_hp,
                              t_log_siteneighbors_4_mp_divided_siteneighbors_3_mp,
                              t_log_siteneighbors_3_hp_divided_siteneighbors_2_hp,
                              t_log_siteneighbors_3_mp_divided_siteneighbors_2_mp,
                              t_log_siteneighbors_2_hp_divided_siteneighbors_1_hp,
                              t_log_siteneighbors_2_mp_divided_siteneighbors_1_mp,
                              t_log_min_siteneighbors_hp,
                              t_log_min_siteneighbors_mp,
                              t_log_max_siteneighbors_hp,
                              t_log_max_siteneighbors_mp,
                              t_log_avg_siteneighbors_hp,
                              t_log_avg_siteneighbors_mp,
                              t_log_siteneighbors_4_hp_siteneighbors_3_hp_pagerank_hp,
                              t_log_siteneighbors_4_hp_siteneighbors_3_mp_pagerank_mp,
                              t_log_siteneighbors_3_hp_siteneighbors_2_hp_pagerank_hp,
                              t_log_siteneighbors_3_hp_siteneighbors_2_mp_pagerank_mp,
                              t_log_siteneighbors_2_hp_siteneighbors_1_hp_pagerank_hp,
                              t_log_siteneighbors_2_hp_siteneighbors_1_hp_pagerank_mp,
                              t_siteneighbors_1_hp_divided_siteneighbors_1_mp,
                              t_siteneighbors_2_hp_divided_siteneighbors_2_mp,
                              t_siteneighbors_3_hp_divided_siteneighbors_3_mp,
                              t_siteneighbors_4_hp_divided_siteneighbors_4_mp,
                              t_log_neighbors_2_hp,
                              t_log_neighbors_3_hp,
                              t_log_neighbors_4_hp,
                              t_log_neighbors_2_mp,
                              t_log_neighbors_3_mp,
                              t_log_neighbors_4_mp,
                              t_log_neighbors_2_hp_divided_pagerank_hp,
                              t_log_neighbors_3_hp_divided_pagerank_hp,
                              t_log_neighbors_4_hp_divided_pagerank_hp,
                              t_log_neighbors_2_mp_divided_pagerank_mp,
                              t_log_neighbors_3_mp_divided_pagerank_mp,
                              t_log_neighbors_4_mp_divided_pagerank_mp,
                              t_log_neighbors_4_hp_divided_neighbors_3_hp,
                              t_log_neighbors_4_mp_divided_neighbors_3_mp,
                              t_log_neighbors_3_hp_divided_neighbors_2_hp,
                              t_log_neighbors_3_mp_divided_neighbors_2_mp,
                              t_log_neighbors_2_hp_divided_indegree_hp,
                              t_log_neighbors_2_mp_divided_indegree_mp,
                              t_log_min_neighbors_hp,
                              t_log_min_neighbors_mp,
                              t_log_max_neighbors_hp,
                              t_log_max_neighbors_mp,
                              t_log_avg_neighbors_hp,
                              t_log_avg_neighbors_mp,
                              t_log_neighbors_4_divided_pagerank_hp,
                              t_log_neighbors_4_divided_pagerank_mp,
                              t_log_neighbors_3_divided_pagerank_hp,
                              t_log_neighbors_3_divided_pagerank_mp,
                              t_log_neighbors_2_divided_pagerank_hp,
                              t_log_neighbors_2_divided_pagerank_mp,
                              t_neighbors_2_hp_divided_neighbors_2_mp,
                              t_neighbors_3_hp_divided_neighbors_3_mp,
                              t_neighbors_4_hp_divided_neighbors_4_mp,
                              t_log_truncatedpagerank_1_hp,
                              t_log_truncatedpagerank_2_hp,
                              t_log_truncatedpagerank_3_hp,
                              t_log_truncatedpagerank_4_hp,
                              t_log_truncatedpagerank_1_mp,
                              t_log_truncatedpagerank_2_mp,
                              t_log_truncatedpagerank_3_mp,
                              t_log_truncatedpagerank_4_mp,
                              t_log_truncatedpagerank_1_hp_divided_pagerank_hp,
                              t_log_truncatedpagerank_2_hp_divided_pagerank_hp,
                              t_log_truncatedpagerank_3_hp_divided_pagerank_hp,
                              t_log_truncatedpagerank_4_hp_divided_pagerank_hp,
                              t_log_truncatedpagerank_1_mp_divided_pagerank_mp,
                              t_log_truncatedpagerank_2_mp_divided_pagerank_mp,
                              t_log_truncatedpagerank_3_mp_divided_pagerank_mp,
                              t_log_truncatedpagerank_4_mp_divided_pagerank_mp,
                              t_truncatedpagerank_4_hp_divided_truncatedpagerank_3_hp,
                              t_truncatedpagerank_4_mp_divided_truncatedpagerank_3_mp,
                              t_truncatedpagerank_3_hp_divided_truncatedpagerank_2_hp,
                              t_truncatedpagerank_3_mp_divided_truncatedpagerank_2_mp,
                              t_truncatedpagerank_2_hp_divided_truncatedpagerank_1_hp,
                              t_truncatedpagerank_2_mp_divided_truncatedpagerank_1_mp,
                              t_log_min_truncatedpagerank_hp,
                              t_log_min_truncatedpagerank_mp,
                              t_log_max_truncatedpagerank_hp,
                              t_log_max_truncatedpagerank_mp,
                              t_log_avg_truncatedpagerank_hp,
                              t_log_avg_truncatedpagerank_mp,
                              #134t_truncatedpagerank_1_mp_divided_truncatedpagerank_1_hp,
                              #135t_truncatedpagerank_2_mp_divided_truncatedpagerank_2_hp,
                              #136t_truncatedpagerank_3_mp_divided_truncatedpagerank_3_hp,
                              #137t_truncatedpagerank_4_mp_divided_truncatedpagerank_4_hp,
                              #t_g
                              #t_b
                              t_HST_1,
                              t_HST_2,
                              t_HST_3,
                              t_HST_4,
                              t_HST_5,
                              t_HST_6,
                              t_HST_7,
                              t_HST_8,
                              t_HST_9,
                              t_HST_10,
                              t_HST_11,
                              t_HST_12,
                              t_HST_13,
                              t_HST_14,
                              t_HST_15,
                              t_HST_16,
                              t_HST_17,
                              t_HST_18,
                              t_HST_19,
                              t_HST_20,
                              t_HST_21,
                              t_HST_22,
                              t_HST_23,
                              t_HST_24,
                              t_HMG_25,
                              t_HMG_26,
                              t_HMG_27,
                              t_HMG_28,
                              t_HMG_29,
                              t_HMG_30,
                              t_HMG_31,
                              t_HMG_32,
                              t_HMG_33,
                              t_HMG_34,
                              t_HMG_35,
                              t_HMG_36,
                              t_HMG_37,
                              t_HMG_38,
                              t_HMG_39,
                              t_HMG_40,
                              t_HMG_41,
                              t_HMG_42,
                              t_HMG_43,
                              t_HMG_44,
                              t_HMG_45,
                              t_HMG_46,
                              t_HMG_47,
                              t_HMG_48,
                              t_AVG_49,
                              t_AVG_50,
                              t_AVG_51,
                              t_AVG_52,
                              t_AVG_53,
                              t_AVG_54,
                              t_AVG_55,
                              t_AVG_56,
                              t_AVG_57,
                              t_AVG_58,
                              t_AVG_59,
                              t_AVG_60,
                              t_AVG_61,
                              t_AVG_62,
                              t_AVG_63,
                              t_AVG_64,
                              t_AVG_65,
                              t_AVG_66,
                              t_AVG_67, 
                              t_AVG_68,
                              t_AVG_69,
                              t_AVG_70,
                              t_AVG_71,
                              t_AVG_72,
                              t_STD_73,
                              t_STD_74,
                              t_STD_75,
                              t_STD_76,
                              t_STD_77,
                              t_STD_78,
                              t_STD_79,
                              t_STD_80,
                              t_STD_81,
                              t_STD_82,
                              t_STD_83,
                              t_STD_84,
                              t_STD_85,
                              t_STD_86,
                              t_STD_87,
                              t_STD_88,
                              t_STD_89,
                              t_STD_90,
                              t_STD_91,
                              t_STD_92,
                              t_STD_93,
                              t_STD_94,
                              t_STD_95,
                              t_STD_96]).T


# In[708]:

#################### establish Tesing Set ######################
with open('WEBSPAM-UK2007-SET2-labels.txt','r') as txtfile:
    test_file = txtfile.readlines()

id_label_test = []
allwords_test = []
label_test = [] 

for i in range(len(test_file)):  
    allwords_test.append(test_file[i].split(' '))

for i in range(len(allwords_test)):
    id_label_test.append(allwords_test[i][0])
    label_test.append(allwords_test[i][1])

for i in range(len(id_label_test)):
    if label_test[i]=='spam':
        label_test[i]=1
    elif label_test[i]=='nonspam':
        label_test[i]=0
    else:
        label_test[i]=2  #undecided
    if id_label_test[i] not in host_id:   #not in content file
        id_label_test[i]=0
        label_test[i]=3
    

#delete content not in file
dellist_test = []
delid_test = []

for i in range(len(label_test)):
    if label_test[i]==3:
        dellist_test.append(label_test[i])
        delid_test.append(id_label_test[i])
for i in dellist_test:
    label_test.remove(i)
for i in delid_test:
    id_label_test.remove(i)

#delete undecided
dellist_test = []
delid_test = []

for i in range(len(label_test)):
    if label_test[i]==2:
        dellist_test.append(label_test[i])
        delid_test.append(id_label_test[i])
for i in dellist_test:
    label_test.remove(i)
for i in delid_test:
    id_label_test.remove(i)


# In[749]:

################# test goodbadrank feature #####################

#for testing set
s_g = []
s_b = []
for ids in id_label_test:
    s_g.append(g[int(ids)])
    s_b.append(b[int(ids)])


# In[712]:

##################### test other link-based features #####################
s_log_indegree_hp = []

s_log_indegree_mp = []

s_log_outdegree_hp = []

s_log_outdegree_mp = []

s_reciprocity_hp = []

s_reciprocity_mp = []

s_log_assortativity_hp = []

s_log_assortativity_mp = []

s_log_avgin_of_out_hp = []

s_log_avgin_of_out_mp = []

s_log_avgin_of_out_hp_times_outdegree_hp = []

s_log_avgin_of_out_hp_times_outdegree_mp = []

s_log_avgout_of_in_hp = []

s_log_avgout_of_in_mp = []

s_log_avgout_of_in_hp_times_indegree_hp = []

s_log_avgout_of_in_hp_times_indegree_mp = []

s_eq_hp_mp = []

s_log_pagerank_hp = []

s_log_pagerank_mp = []

s_log_indegree_hp_divided_pagerank_hp = []

s_log_indegree_mp_divided_pagerank_mp = []

s_log_outdegree_hp_divided_pagerank_hp = []

s_log_outdegree_mp_divided_pagerank_mp = []

s_log_prsigma_hp = []

s_log_prsigma_mp = []

s_log_prsigma_hp_divided_pagerank_hp = []

s_log_prsigma_mp_divided_pagerank_mp = []

s_pagerank_mp_divided_pagerank_hp = []

s_log_trustrank_hp = []

s_log_trustrank_mp = []

s_log_trustrank_hp_divided_pagerank_hp = []

s_log_trustrank_mp_divided_pagerank_mp = []

s_log_trustrank_hp_divided_indegree_hp = []

s_log_trustrank_mp_divided_indegree_mp = []

s_trustrank_mp_divided_trustrank_hp = []

s_log_siteneighbors_1_hp = []

s_log_siteneighbors_2_hp = []

s_log_siteneighbors_3_hp = []

s_log_siteneighbors_4_hp = []

s_log_siteneighbors_1_mp = []

s_log_siteneighbors_2_mp = []

s_log_siteneighbors_3_mp = []

s_log_siteneighbors_4_mp = []

s_log_siteneighbors_1_hp_divided_pagerank_hp = []

s_log_siteneighbors_2_hp_divided_pagerank_hp = []

s_log_siteneighbors_3_hp_divided_pagerank_hp = []

s_log_siteneighbors_4_hp_divided_pagerank_hp = []

s_log_siteneighbors_1_mp_divided_pagerank_mp = []

s_log_siteneighbors_2_mp_divided_pagerank_mp = []

s_log_siteneighbors_3_mp_divided_pagerank_mp = []

s_log_siteneighbors_4_mp_divided_pagerank_mp = []

s_log_siteneighbors_4_hp_divided_siteneighbors_3_hp = []

s_log_siteneighbors_4_mp_divided_siteneighbors_3_mp = []

s_log_siteneighbors_3_hp_divided_siteneighbors_2_hp = []

s_log_siteneighbors_3_mp_divided_siteneighbors_2_mp = []

s_log_siteneighbors_2_hp_divided_siteneighbors_1_hp = []

s_log_siteneighbors_2_mp_divided_siteneighbors_1_mp = []

s_log_min_siteneighbors_hp = []

s_log_min_siteneighbors_mp = []

s_log_max_siteneighbors_hp = []

s_log_max_siteneighbors_mp = []

s_log_avg_siteneighbors_hp = []

s_log_avg_siteneighbors_mp = []

s_log_siteneighbors_4_hp_siteneighbors_3_hp_pagerank_hp = []

s_log_siteneighbors_4_hp_siteneighbors_3_mp_pagerank_mp = []

s_log_siteneighbors_3_hp_siteneighbors_2_hp_pagerank_hp = []

s_log_siteneighbors_3_hp_siteneighbors_2_mp_pagerank_mp = []

s_log_siteneighbors_2_hp_siteneighbors_1_hp_pagerank_hp = []

s_log_siteneighbors_2_hp_siteneighbors_1_hp_pagerank_mp = []

s_siteneighbors_1_hp_divided_siteneighbors_1_mp = []

s_siteneighbors_2_hp_divided_siteneighbors_2_mp = []

s_siteneighbors_3_hp_divided_siteneighbors_3_mp = []

s_siteneighbors_4_hp_divided_siteneighbors_4_mp = []

s_log_neighbors_2_hp = []

s_log_neighbors_3_hp = []

s_log_neighbors_4_hp = []

s_log_neighbors_2_mp = []

s_log_neighbors_3_mp = []

s_log_neighbors_4_mp = []

s_log_neighbors_2_hp_divided_pagerank_hp = []

s_log_neighbors_3_hp_divided_pagerank_hp = []

s_log_neighbors_4_hp_divided_pagerank_hp = []

s_log_neighbors_2_mp_divided_pagerank_mp = []

s_log_neighbors_3_mp_divided_pagerank_mp = []

s_log_neighbors_4_mp_divided_pagerank_mp = []

s_log_neighbors_4_hp_divided_neighbors_3_hp = []

s_log_neighbors_4_mp_divided_neighbors_3_mp = []

s_log_neighbors_3_hp_divided_neighbors_2_hp = []

s_log_neighbors_3_mp_divided_neighbors_2_mp = []

s_log_neighbors_2_hp_divided_indegree_hp = []

s_log_neighbors_2_mp_divided_indegree_mp = []

s_log_min_neighbors_hp = []

s_log_min_neighbors_mp = []

s_log_max_neighbors_hp = []

s_log_max_neighbors_mp = []

s_log_avg_neighbors_hp = []

s_log_avg_neighbors_mp = []

s_log_neighbors_4_divided_pagerank_hp = []

s_log_neighbors_4_divided_pagerank_mp = []

s_log_neighbors_3_divided_pagerank_hp = []

s_log_neighbors_3_divided_pagerank_mp = []

s_log_neighbors_2_divided_pagerank_hp = []

s_log_neighbors_2_divided_pagerank_mp = []

s_neighbors_2_hp_divided_neighbors_2_mp = []

s_neighbors_3_hp_divided_neighbors_3_mp = []

s_neighbors_4_hp_divided_neighbors_4_mp = []

s_log_truncatedpagerank_1_hp = []

s_log_truncatedpagerank_2_hp = []

s_log_truncatedpagerank_3_hp = []

s_log_truncatedpagerank_4_hp = []

s_log_truncatedpagerank_1_mp = []

s_log_truncatedpagerank_2_mp = []

s_log_truncatedpagerank_3_mp = []

s_log_truncatedpagerank_4_mp = []

s_log_truncatedpagerank_1_hp_divided_pagerank_hp = []

s_log_truncatedpagerank_2_hp_divided_pagerank_hp = []

s_log_truncatedpagerank_3_hp_divided_pagerank_hp = []

s_log_truncatedpagerank_4_hp_divided_pagerank_hp = []

s_log_truncatedpagerank_1_mp_divided_pagerank_mp = []

s_log_truncatedpagerank_2_mp_divided_pagerank_mp = []

s_log_truncatedpagerank_3_mp_divided_pagerank_mp = []

s_log_truncatedpagerank_4_mp_divided_pagerank_mp = []

s_truncatedpagerank_4_hp_divided_truncatedpagerank_3_hp = []

s_truncatedpagerank_4_mp_divided_truncatedpagerank_3_mp = []

s_truncatedpagerank_3_hp_divided_truncatedpagerank_2_hp = []

s_truncatedpagerank_3_mp_divided_truncatedpagerank_2_mp = []

s_truncatedpagerank_2_hp_divided_truncatedpagerank_1_hp = []

s_truncatedpagerank_2_mp_divided_truncatedpagerank_1_mp = []

s_log_min_truncatedpagerank_hp = []

s_log_min_truncatedpagerank_mp = []

s_log_max_truncatedpagerank_hp = []

s_log_max_truncatedpagerank_mp = []

s_log_avg_truncatedpagerank_hp = []

s_log_avg_truncatedpagerank_mp = []

s_truncatedpagerank_1_mp_divided_truncatedpagerank_1_hp = []

s_truncatedpagerank_2_mp_divided_truncatedpagerank_2_hp = []

s_truncatedpagerank_3_mp_divided_truncatedpagerank_3_hp = []

s_truncatedpagerank_4_mp_divided_truncatedpagerank_4_hp = []


# In[713]:

for ids in id_label_test:
    
    s_log_indegree_hp.append(log_indegree_hp[int(ids)])
    
    s_log_indegree_mp.append(log_indegree_mp[int(ids)])
    
    s_log_outdegree_hp.append(log_outdegree_hp[int(ids)])
    
    s_log_outdegree_mp.append(log_outdegree_mp[int(ids)])
    
    s_reciprocity_hp.append(reciprocity_hp[int(ids)])
    
    s_reciprocity_mp.append(reciprocity_mp[int(ids)])
    
    s_log_assortativity_hp.append(log_assortativity_hp[int(ids)])
    
    s_log_assortativity_mp.append(log_assortativity_mp[int(ids)])
    
    s_log_avgin_of_out_hp.append(log_avgin_of_out_hp[int(ids)])
    
    s_log_avgin_of_out_mp.append(log_avgin_of_out_mp[int(ids)]) 
    
    s_log_avgin_of_out_hp_times_outdegree_hp.append(log_avgin_of_out_hp_times_outdegree_hp[int(ids)]) 
    
    s_log_avgin_of_out_hp_times_outdegree_mp.append(log_avgin_of_out_hp_times_outdegree_mp[int(ids)]) 
    
    s_log_avgout_of_in_hp.append(log_avgout_of_in_hp[int(ids)]) 
    
    s_log_avgout_of_in_mp.append(log_avgout_of_in_mp[int(ids)]) 
    
    s_log_avgout_of_in_hp_times_indegree_hp.append(log_avgout_of_in_hp_times_indegree_hp[int(ids)])
    
    s_log_avgout_of_in_hp_times_indegree_mp.append(log_avgout_of_in_hp_times_indegree_mp[int(ids)])
    
    s_eq_hp_mp.append(eq_hp_mp[int(ids)])
    
    s_log_pagerank_hp.append(log_pagerank_hp[int(ids)])
    
    s_log_pagerank_mp.append(log_pagerank_mp[int(ids)])
    
    s_log_indegree_hp_divided_pagerank_hp.append(log_indegree_hp_divided_pagerank_hp[int(ids)])
    
    s_log_indegree_mp_divided_pagerank_mp.append(log_indegree_mp_divided_pagerank_mp[int(ids)])
    
    s_log_outdegree_hp_divided_pagerank_hp.append(log_outdegree_hp_divided_pagerank_hp[int(ids)])
    
    s_log_outdegree_mp_divided_pagerank_mp.append(log_outdegree_mp_divided_pagerank_mp[int(ids)])
    
    s_log_prsigma_hp.append(log_prsigma_hp[int(ids)])
    
    s_log_prsigma_mp.append(log_prsigma_mp[int(ids)])
    
    s_log_prsigma_hp_divided_pagerank_hp.append(log_prsigma_hp_divided_pagerank_hp[int(ids)])
    
    s_log_prsigma_mp_divided_pagerank_mp.append(log_prsigma_mp_divided_pagerank_mp[int(ids)])
    
    s_pagerank_mp_divided_pagerank_hp.append(pagerank_mp_divided_pagerank_hp[int(ids)])
    
    s_log_trustrank_hp.append(log_trustrank_hp[int(ids)])
    
    s_log_trustrank_mp.append(log_trustrank_mp[int(ids)])
    
    s_log_trustrank_hp_divided_pagerank_hp.append(log_trustrank_hp_divided_pagerank_hp[int(ids)])
    
    s_log_trustrank_mp_divided_pagerank_mp.append(log_trustrank_mp_divided_pagerank_mp[int(ids)])
    
    s_log_trustrank_hp_divided_indegree_hp.append(log_trustrank_hp_divided_indegree_hp[int(ids)])
    
    s_log_trustrank_mp_divided_indegree_mp.append(log_trustrank_mp_divided_indegree_mp[int(ids)])
    
    s_trustrank_mp_divided_trustrank_hp.append(trustrank_mp_divided_trustrank_hp[int(ids)])
    
    s_log_siteneighbors_1_hp.append(log_siteneighbors_1_hp[int(ids)])
    
    s_log_siteneighbors_2_hp.append(log_siteneighbors_2_hp[int(ids)])
    
    s_log_siteneighbors_3_hp.append(log_siteneighbors_3_hp[int(ids)])
    
    s_log_siteneighbors_4_hp.append(log_siteneighbors_4_hp[int(ids)])
    
    s_log_siteneighbors_1_mp.append(log_siteneighbors_1_mp[int(ids)])
    
    s_log_siteneighbors_2_mp.append(log_siteneighbors_2_mp[int(ids)])
    
    s_log_siteneighbors_3_mp.append(log_siteneighbors_3_mp[int(ids)])
    
    s_log_siteneighbors_4_mp.append(log_siteneighbors_4_mp[int(ids)])
    
    s_log_siteneighbors_1_hp_divided_pagerank_hp.append(log_siteneighbors_1_hp_divided_pagerank_hp[int(ids)])
    
    s_log_siteneighbors_2_hp_divided_pagerank_hp.append(log_siteneighbors_2_hp_divided_pagerank_hp[int(ids)])
    
    s_log_siteneighbors_3_hp_divided_pagerank_hp.append(log_siteneighbors_3_hp_divided_pagerank_hp[int(ids)])
    
    s_log_siteneighbors_4_hp_divided_pagerank_hp.append(log_siteneighbors_4_hp_divided_pagerank_hp[int(ids)])
    
    s_log_siteneighbors_1_mp_divided_pagerank_mp.append(log_siteneighbors_1_mp_divided_pagerank_mp[int(ids)])
    
    s_log_siteneighbors_2_mp_divided_pagerank_mp.append(log_siteneighbors_2_mp_divided_pagerank_mp[int(ids)])
    
    s_log_siteneighbors_3_mp_divided_pagerank_mp.append(log_siteneighbors_3_mp_divided_pagerank_mp[int(ids)])
    
    s_log_siteneighbors_4_mp_divided_pagerank_mp.append(log_siteneighbors_4_mp_divided_pagerank_mp[int(ids)])
    
    s_log_siteneighbors_4_hp_divided_siteneighbors_3_hp.append(log_siteneighbors_4_hp_divided_siteneighbors_3_hp[int(ids)])
    
    s_log_siteneighbors_4_mp_divided_siteneighbors_3_mp.append(log_siteneighbors_4_mp_divided_siteneighbors_3_mp[int(ids)])
    
    s_log_siteneighbors_3_hp_divided_siteneighbors_2_hp.append(log_siteneighbors_3_hp_divided_siteneighbors_2_hp[int(ids)])
    
    s_log_siteneighbors_3_mp_divided_siteneighbors_2_mp.append(log_siteneighbors_3_mp_divided_siteneighbors_2_mp[int(ids)])
    
    s_log_siteneighbors_2_hp_divided_siteneighbors_1_hp.append(log_siteneighbors_2_hp_divided_siteneighbors_1_hp[int(ids)])
    
    s_log_siteneighbors_2_mp_divided_siteneighbors_1_mp.append(log_siteneighbors_2_mp_divided_siteneighbors_1_mp[int(ids)])
    
    s_log_min_siteneighbors_hp.append(log_min_siteneighbors_hp[int(ids)])
    
    s_log_min_siteneighbors_mp.append(log_min_siteneighbors_mp[int(ids)])
    
    s_log_max_siteneighbors_hp.append(log_max_siteneighbors_hp[int(ids)])
    
    s_log_max_siteneighbors_mp.append(log_max_siteneighbors_mp[int(ids)])
    
    s_log_avg_siteneighbors_hp.append(log_avg_siteneighbors_hp[int(ids)])
    
    s_log_avg_siteneighbors_mp.append(log_avg_siteneighbors_mp[int(ids)])
    
    s_log_siteneighbors_4_hp_siteneighbors_3_hp_pagerank_hp.append(log_siteneighbors_4_hp_siteneighbors_3_hp_pagerank_hp[int(ids)])
    
    s_log_siteneighbors_4_hp_siteneighbors_3_mp_pagerank_mp.append(log_siteneighbors_4_hp_siteneighbors_3_mp_pagerank_mp[int(ids)])
    
    s_log_siteneighbors_3_hp_siteneighbors_2_hp_pagerank_hp.append(log_siteneighbors_3_hp_siteneighbors_2_hp_pagerank_hp[int(ids)])
    
    s_log_siteneighbors_3_hp_siteneighbors_2_mp_pagerank_mp.append(log_siteneighbors_3_hp_siteneighbors_2_mp_pagerank_mp[int(ids)])
    
    s_log_siteneighbors_2_hp_siteneighbors_1_hp_pagerank_hp.append(log_siteneighbors_2_hp_siteneighbors_1_hp_pagerank_hp[int(ids)])
    
    s_log_siteneighbors_2_hp_siteneighbors_1_hp_pagerank_mp.append(log_siteneighbors_2_hp_siteneighbors_1_hp_pagerank_mp[int(ids)])
    
    s_siteneighbors_1_hp_divided_siteneighbors_1_mp.append(siteneighbors_1_hp_divided_siteneighbors_1_mp[int(ids)])
    #start at 71
    s_siteneighbors_2_hp_divided_siteneighbors_2_mp.append(siteneighbors_2_hp_divided_siteneighbors_2_mp[int(ids)])
    
    s_siteneighbors_3_hp_divided_siteneighbors_3_mp.append(siteneighbors_3_hp_divided_siteneighbors_3_mp[int(ids)])

    s_siteneighbors_4_hp_divided_siteneighbors_4_mp.append(siteneighbors_4_hp_divided_siteneighbors_4_mp[int(ids)])

    s_log_neighbors_2_hp.append(log_neighbors_2_hp[int(ids)])

    s_log_neighbors_3_hp.append(log_neighbors_3_hp[int(ids)])

    s_log_neighbors_4_hp.append(log_neighbors_4_hp[int(ids)])

    s_log_neighbors_2_mp.append(log_neighbors_2_mp[int(ids)])

    s_log_neighbors_3_mp.append(log_neighbors_3_mp[int(ids)])

    s_log_neighbors_4_mp.append(log_neighbors_4_mp[int(ids)])

    s_log_neighbors_2_hp_divided_pagerank_hp.append(log_neighbors_2_hp_divided_pagerank_hp[int(ids)])

    s_log_neighbors_3_hp_divided_pagerank_hp.append(log_neighbors_3_hp_divided_pagerank_hp[int(ids)])

    s_log_neighbors_4_hp_divided_pagerank_hp.append(log_neighbors_4_hp_divided_pagerank_hp[int(ids)])

    s_log_neighbors_2_mp_divided_pagerank_mp.append(log_neighbors_2_mp_divided_pagerank_mp[int(ids)])

    s_log_neighbors_3_mp_divided_pagerank_mp.append(log_neighbors_3_mp_divided_pagerank_mp[int(ids)])

    s_log_neighbors_4_mp_divided_pagerank_mp.append(log_neighbors_4_mp_divided_pagerank_mp[int(ids)])

    s_log_neighbors_4_hp_divided_neighbors_3_hp.append(log_neighbors_4_hp_divided_neighbors_3_hp[int(ids)])

    s_log_neighbors_4_mp_divided_neighbors_3_mp.append(log_neighbors_4_mp_divided_neighbors_3_mp[int(ids)])

    s_log_neighbors_3_hp_divided_neighbors_2_hp.append(log_neighbors_3_hp_divided_neighbors_2_hp[int(ids)])

    s_log_neighbors_3_mp_divided_neighbors_2_mp.append(log_neighbors_3_mp_divided_neighbors_2_mp[int(ids)])

    s_log_neighbors_2_hp_divided_indegree_hp.append(log_neighbors_2_hp_divided_indegree_hp[int(ids)])

    s_log_neighbors_2_mp_divided_indegree_mp.append(log_neighbors_2_mp_divided_indegree_mp[int(ids)])

    s_log_min_neighbors_hp.append(log_min_neighbors_hp[int(ids)])

    s_log_min_neighbors_mp.append(log_min_neighbors_mp[int(ids)])

    s_log_max_neighbors_hp.append(log_max_neighbors_hp[int(ids)])

    s_log_max_neighbors_mp.append(log_max_neighbors_mp[int(ids)])
    
    s_log_avg_neighbors_hp.append(log_avg_neighbors_hp[int(ids)])
    
    s_log_avg_neighbors_mp.append(log_avg_neighbors_mp[int(ids)])

    s_log_neighbors_4_divided_pagerank_hp.append(log_neighbors_4_divided_pagerank_hp[int(ids)])

    s_log_neighbors_4_divided_pagerank_mp.append(log_neighbors_4_divided_pagerank_mp[int(ids)])

    s_log_neighbors_3_divided_pagerank_hp.append(log_neighbors_3_divided_pagerank_hp[int(ids)])

    s_log_neighbors_3_divided_pagerank_mp.append(log_neighbors_3_divided_pagerank_mp[int(ids)])

    s_log_neighbors_2_divided_pagerank_hp.append(log_neighbors_2_divided_pagerank_hp[int(ids)])

    s_log_neighbors_2_divided_pagerank_mp.append(log_neighbors_2_divided_pagerank_mp[int(ids)])

    s_neighbors_2_hp_divided_neighbors_2_mp.append(neighbors_2_hp_divided_neighbors_2_mp[int(ids)])

    s_neighbors_3_hp_divided_neighbors_3_mp.append(neighbors_3_hp_divided_neighbors_3_mp[int(ids)])

    s_neighbors_4_hp_divided_neighbors_4_mp.append(neighbors_4_hp_divided_neighbors_4_mp[int(ids)])

    s_log_truncatedpagerank_1_hp.append(log_truncatedpagerank_1_hp[int(ids)])

    s_log_truncatedpagerank_2_hp.append(log_truncatedpagerank_2_hp[int(ids)])

    s_log_truncatedpagerank_3_hp.append(log_truncatedpagerank_3_hp[int(ids)])

    s_log_truncatedpagerank_4_hp.append(log_truncatedpagerank_4_hp[int(ids)])

    s_log_truncatedpagerank_1_mp.append(log_truncatedpagerank_1_mp[int(ids)])

    s_log_truncatedpagerank_2_mp.append(log_truncatedpagerank_2_mp[int(ids)])

    s_log_truncatedpagerank_3_mp.append(log_truncatedpagerank_3_mp[int(ids)])

    s_log_truncatedpagerank_4_mp.append(log_truncatedpagerank_4_mp[int(ids)])

    s_log_truncatedpagerank_1_hp_divided_pagerank_hp.append(log_truncatedpagerank_1_hp_divided_pagerank_hp[int(ids)])

    s_log_truncatedpagerank_2_hp_divided_pagerank_hp.append(log_truncatedpagerank_2_hp_divided_pagerank_hp[int(ids)])

    s_log_truncatedpagerank_3_hp_divided_pagerank_hp.append(log_truncatedpagerank_3_hp_divided_pagerank_hp[int(ids)])

    s_log_truncatedpagerank_4_hp_divided_pagerank_hp.append(log_truncatedpagerank_4_hp_divided_pagerank_hp[int(ids)])

    s_log_truncatedpagerank_1_mp_divided_pagerank_mp.append(log_truncatedpagerank_1_mp_divided_pagerank_mp[int(ids)])

    s_log_truncatedpagerank_2_mp_divided_pagerank_mp.append(log_truncatedpagerank_2_mp_divided_pagerank_mp[int(ids)])

    s_log_truncatedpagerank_3_mp_divided_pagerank_mp.append(log_truncatedpagerank_3_mp_divided_pagerank_mp[int(ids)])

    s_log_truncatedpagerank_4_mp_divided_pagerank_mp.append(log_truncatedpagerank_4_mp_divided_pagerank_mp[int(ids)])

    s_truncatedpagerank_4_hp_divided_truncatedpagerank_3_hp.append(truncatedpagerank_4_hp_divided_truncatedpagerank_3_hp[int(ids)])

    s_truncatedpagerank_4_mp_divided_truncatedpagerank_3_mp.append(truncatedpagerank_4_mp_divided_truncatedpagerank_3_mp[int(ids)])

    s_truncatedpagerank_3_hp_divided_truncatedpagerank_2_hp.append(truncatedpagerank_3_hp_divided_truncatedpagerank_2_hp[int(ids)])

    s_truncatedpagerank_3_mp_divided_truncatedpagerank_2_mp.append(truncatedpagerank_3_mp_divided_truncatedpagerank_2_mp[int(ids)])

    s_truncatedpagerank_2_hp_divided_truncatedpagerank_1_hp.append(truncatedpagerank_2_hp_divided_truncatedpagerank_1_hp[int(ids)])

    s_truncatedpagerank_2_mp_divided_truncatedpagerank_1_mp.append(truncatedpagerank_2_mp_divided_truncatedpagerank_1_mp[int(ids)])

    s_log_min_truncatedpagerank_hp.append(log_min_truncatedpagerank_hp[int(ids)])

    s_log_min_truncatedpagerank_mp.append(log_min_truncatedpagerank_mp[int(ids)])

    s_log_max_truncatedpagerank_hp.append(log_max_truncatedpagerank_hp[int(ids)])

    s_log_max_truncatedpagerank_mp.append(log_max_truncatedpagerank_mp[int(ids)])

    s_log_avg_truncatedpagerank_hp.append(log_avg_truncatedpagerank_hp[int(ids)])

    s_log_avg_truncatedpagerank_mp.append(log_avg_truncatedpagerank_mp[int(ids)])

    s_truncatedpagerank_1_mp_divided_truncatedpagerank_1_hp.append(truncatedpagerank_1_mp_divided_truncatedpagerank_1_hp[int(ids)])

    s_truncatedpagerank_2_mp_divided_truncatedpagerank_2_hp.append(truncatedpagerank_2_mp_divided_truncatedpagerank_2_hp[int(ids)])

    s_truncatedpagerank_3_mp_divided_truncatedpagerank_3_hp.append(truncatedpagerank_3_mp_divided_truncatedpagerank_3_hp[int(ids)])
    
    s_truncatedpagerank_4_mp_divided_truncatedpagerank_4_hp.append(truncatedpagerank_4_mp_divided_truncatedpagerank_4_hp[int(ids)])


# In[819]:

################### test content-based features ###################
s_HST_1 = []

s_HST_2 = []

s_HST_3 = []

s_HST_4 = []

s_HST_5 = []

s_HST_6 = []

s_HST_7 = []

s_HST_8 = []

s_HST_9 = []

s_HST_10 = []

s_HST_11 = []

s_HST_12 = []

s_HST_13 = []

s_HST_14 = []

s_HST_15 = []

s_HST_16 = []

s_HST_17 = []

s_HST_18 = []

s_HST_19 = []

s_HST_20 = []

s_HST_21 = []

s_HST_22 = []

s_HST_23 = []

s_HST_24 = []

s_HMG_25 = []

s_HMG_26 = []

s_HMG_27 = []

s_HMG_28 = []

s_HMG_29 = []

s_HMG_30 = []

s_HMG_31 = []

s_HMG_32 = []

s_HMG_33 = []

s_HMG_34 = []

s_HMG_35 = []

s_HMG_36 = []

s_HMG_37 = []

s_HMG_38 = []

s_HMG_39 = []

s_HMG_40 = []

s_HMG_41 = []

s_HMG_42 = []

s_HMG_43 = []

s_HMG_44 = []

s_HMG_45 = []

s_HMG_46 = []

s_HMG_47 = []

s_HMG_48 = []

s_AVG_49 = []

s_AVG_50 = []

s_AVG_51 = []

s_AVG_52 = []

s_AVG_53 = []

s_AVG_54 = []

s_AVG_55 = []

s_AVG_56 = []

s_AVG_57 = []

s_AVG_58 = []

s_AVG_59 = []

s_AVG_60 = []

s_AVG_61 = []

s_AVG_62 = []

s_AVG_63 = []

s_AVG_64 = []

s_AVG_65 = []

s_AVG_66 = []

s_AVG_67 = []

s_AVG_68 = []

s_AVG_69 = []

s_AVG_70 = []

s_AVG_71 = []

s_AVG_72 = []

s_STD_73 = []

s_STD_74 = []

s_STD_75 = []

s_STD_76 = []

s_STD_77 = []

s_STD_78 = []

s_STD_79 = []

s_STD_80 = []

s_STD_81 = []

s_STD_82 = []

s_STD_83 = []

s_STD_84 = []

s_STD_85 = []

s_STD_86 = []

s_STD_87 = []

s_STD_88 = []

s_STD_89 = []

s_STD_90 = []

s_STD_91 = []

s_STD_92 = []

s_STD_93 = []

s_STD_94 = []

s_STD_95 = []

s_STD_96 = []


# In[820]:

k=0
for ids in id_label_test:
    k=host_id.index(ids)
    s_HST_1.append(HST_1[k])
    s_HST_2.append(HST_2[k])
    s_HST_3.append(HST_3[k])
    s_HST_4.append(HST_4[k])
    s_HST_5.append(HST_5[k])
    s_HST_6.append(HST_6[k])
    s_HST_7.append(HST_7[k])
    s_HST_8.append(HST_8[k])
    s_HST_9.append(HST_9[k])
    s_HST_10.append(HST_10[k])
    s_HST_11.append(HST_11[k])
    s_HST_12.append(HST_12[k])
    s_HST_13.append(HST_13[k])
    s_HST_14.append(HST_14[k])
    s_HST_15.append(HST_15[k])
    s_HST_16.append(HST_16[k])
    s_HST_17.append(HST_17[k])
    s_HST_18.append(HST_18[k])
    s_HST_19.append(HST_19[k])
    s_HST_20.append(HST_20[k])
    s_HST_21.append(HST_21[k])
    s_HST_22.append(HST_22[k])
    s_HST_23.append(HST_23[k])
    s_HST_24.append(HST_24[k])
    s_HMG_25.append(HMG_25[k])
    s_HMG_26.append(HMG_26[k])
    s_HMG_27.append(HMG_27[k])
    s_HMG_28.append(HMG_28[k])
    s_HMG_29.append(HMG_29[k])
    s_HMG_30.append(HMG_30[k])
    s_HMG_31.append(HMG_31[k])
    s_HMG_32.append(HMG_32[k])
    s_HMG_33.append(HMG_33[k])
    s_HMG_34.append(HMG_34[k])
    s_HMG_35.append(HMG_35[k])
    s_HMG_36.append(HMG_36[k])
    s_HMG_37.append(HMG_37[k])
    s_HMG_38.append(HMG_38[k])
    s_HMG_39.append(HMG_39[k])
    s_HMG_40.append(HMG_40[k])
    s_HMG_41.append(HMG_41[k])
    s_HMG_42.append(HMG_42[k])
    s_HMG_43.append(HMG_43[k])
    s_HMG_44.append(HMG_44[k])
    s_HMG_45.append(HMG_45[k])
    s_HMG_46.append(HMG_46[k])
    s_HMG_47.append(HMG_47[k])
    s_HMG_48.append(HMG_48[k])
    s_AVG_49.append(AVG_49[k])
    s_AVG_50.append(AVG_50[k])
    s_AVG_51.append(AVG_51[k])
    s_AVG_52.append(AVG_52[k])
    s_AVG_53.append(AVG_53[k])
    s_AVG_54.append(AVG_54[k])
    s_AVG_55.append(AVG_55[k])
    s_AVG_56.append(AVG_56[k])
    s_AVG_57.append(AVG_57[k])
    s_AVG_58.append(AVG_58[k])
    s_AVG_59.append(AVG_59[k])
    s_AVG_60.append(AVG_60[k])
    s_AVG_61.append(AVG_61[k])
    s_AVG_62.append(AVG_62[k])
    s_AVG_63.append(AVG_63[k])
    s_AVG_64.append(AVG_64[k])
    s_AVG_65.append(AVG_65[k])
    s_AVG_66.append(AVG_66[k])
    s_AVG_67.append(AVG_67[k])
    s_AVG_68.append(AVG_68[k])
    s_AVG_69.append(AVG_69[k])
    s_AVG_70.append(AVG_70[k])
    s_AVG_71.append(AVG_71[k])
    s_AVG_72.append(AVG_72[k])
    s_STD_73.append(STD_73[k])
    s_STD_74.append(STD_74[k])
    s_STD_75.append(STD_75[k])
    s_STD_76.append(STD_76[k])
    s_STD_77.append(STD_77[k])
    s_STD_78.append(STD_78[k])
    s_STD_79.append(STD_79[k])
    s_STD_80.append(STD_80[k])
    s_STD_81.append(STD_81[k])
    s_STD_82.append(STD_82[k])
    s_STD_83.append(STD_83[k])
    s_STD_84.append(STD_84[k])
    s_STD_85.append(STD_85[k])
    s_STD_86.append(STD_86[k])
    s_STD_87.append(STD_87[k])
    s_STD_88.append(STD_88[k])
    s_STD_89.append(STD_89[k])
    s_STD_90.append(STD_90[k])
    s_STD_91.append(STD_91[k])
    s_STD_92.append(STD_92[k])
    s_STD_93.append(STD_93[k])
    s_STD_94.append(STD_94[k])
    s_STD_95.append(STD_95[k])
    s_STD_96.append(STD_96[k])


# In[901]:

#convert tesing features into array
testing_features = np.array([s_log_indegree_hp,
                             s_log_indegree_mp,
                             s_log_outdegree_hp,
                             s_log_outdegree_mp,
                             #4s_reciprocity_hp,
                             s_reciprocity_mp,
                             s_log_assortativity_hp,
                             s_log_assortativity_mp,
                             s_log_avgin_of_out_hp,
                             s_log_avgin_of_out_mp,
                             s_log_avgin_of_out_hp_times_outdegree_hp,
                             s_log_avgin_of_out_hp_times_outdegree_mp,
                             s_log_avgout_of_in_hp,
                             s_log_avgout_of_in_mp,
                             s_log_avgout_of_in_hp_times_indegree_hp,
                             s_log_avgout_of_in_hp_times_indegree_mp,
                             #16s_eq_hp_mp,
                             s_log_pagerank_hp,
                             s_log_pagerank_mp,
                             s_log_indegree_hp_divided_pagerank_hp,
                             s_log_indegree_mp_divided_pagerank_mp,
                             s_log_outdegree_hp_divided_pagerank_hp,
                             s_log_outdegree_mp_divided_pagerank_mp,
                             s_log_prsigma_hp,
                             s_log_prsigma_mp,
                             s_log_prsigma_hp_divided_pagerank_hp,
                             s_log_prsigma_mp_divided_pagerank_mp,
                             #27s_pagerank_mp_divided_pagerank_hp,
                             s_log_trustrank_hp,
                             s_log_trustrank_mp,
                             s_log_trustrank_hp_divided_pagerank_hp,
                             s_log_trustrank_mp_divided_pagerank_mp,
                             s_log_trustrank_hp_divided_indegree_hp,
                             s_log_trustrank_mp_divided_indegree_mp,
                             #34s_trustrank_mp_divided_trustrank_hp,
                             s_log_siteneighbors_1_hp,
                             s_log_siteneighbors_2_hp,
                             s_log_siteneighbors_3_hp,
                             s_log_siteneighbors_4_hp,
                             s_log_siteneighbors_1_mp,
                             s_log_siteneighbors_2_mp,
                             s_log_siteneighbors_3_mp,
                             s_log_siteneighbors_4_mp,
                             s_log_siteneighbors_1_hp_divided_pagerank_hp,
                             s_log_siteneighbors_2_hp_divided_pagerank_hp,
                             s_log_siteneighbors_3_hp_divided_pagerank_hp,
                             s_log_siteneighbors_4_hp_divided_pagerank_hp,
                             s_log_siteneighbors_1_mp_divided_pagerank_mp,
                             s_log_siteneighbors_2_mp_divided_pagerank_mp,
                             s_log_siteneighbors_3_mp_divided_pagerank_mp,
                             s_log_siteneighbors_4_mp_divided_pagerank_mp,
                             s_log_siteneighbors_4_hp_divided_siteneighbors_3_hp,
                             s_log_siteneighbors_4_mp_divided_siteneighbors_3_mp,
                             s_log_siteneighbors_3_hp_divided_siteneighbors_2_hp,
                             s_log_siteneighbors_3_mp_divided_siteneighbors_2_mp,
                             s_log_siteneighbors_2_hp_divided_siteneighbors_1_hp,
                             s_log_siteneighbors_2_mp_divided_siteneighbors_1_mp,
                             s_log_min_siteneighbors_hp,
                             s_log_min_siteneighbors_mp,
                             s_log_max_siteneighbors_hp,
                             s_log_max_siteneighbors_mp,
                             s_log_avg_siteneighbors_hp,
                             s_log_avg_siteneighbors_mp,
                             s_log_siteneighbors_4_hp_siteneighbors_3_hp_pagerank_hp,
                             s_log_siteneighbors_4_hp_siteneighbors_3_mp_pagerank_mp,
                             s_log_siteneighbors_3_hp_siteneighbors_2_hp_pagerank_hp,
                             s_log_siteneighbors_3_hp_siteneighbors_2_mp_pagerank_mp,
                             s_log_siteneighbors_2_hp_siteneighbors_1_hp_pagerank_hp,
                             s_log_siteneighbors_2_hp_siteneighbors_1_hp_pagerank_mp,
                             s_siteneighbors_1_hp_divided_siteneighbors_1_mp,
                             s_siteneighbors_2_hp_divided_siteneighbors_2_mp,
                             s_siteneighbors_3_hp_divided_siteneighbors_3_mp,
                             s_siteneighbors_4_hp_divided_siteneighbors_4_mp,
                             s_log_neighbors_2_hp,
                             s_log_neighbors_3_hp,
                             s_log_neighbors_4_hp,
                             s_log_neighbors_2_mp,
                             s_log_neighbors_3_mp,
                             s_log_neighbors_4_mp,
                             s_log_neighbors_2_hp_divided_pagerank_hp,
                             s_log_neighbors_3_hp_divided_pagerank_hp,
                             s_log_neighbors_4_hp_divided_pagerank_hp,
                             s_log_neighbors_2_mp_divided_pagerank_mp,
                             s_log_neighbors_3_mp_divided_pagerank_mp,
                             s_log_neighbors_4_mp_divided_pagerank_mp,
                             s_log_neighbors_4_hp_divided_neighbors_3_hp,
                             s_log_neighbors_4_mp_divided_neighbors_3_mp,
                             s_log_neighbors_3_hp_divided_neighbors_2_hp,
                             s_log_neighbors_3_mp_divided_neighbors_2_mp,
                             s_log_neighbors_2_hp_divided_indegree_hp,
                             s_log_neighbors_2_mp_divided_indegree_mp,
                             s_log_min_neighbors_hp,
                             s_log_min_neighbors_mp,
                             s_log_max_neighbors_hp,
                             s_log_max_neighbors_mp,
                             s_log_avg_neighbors_hp,
                             s_log_avg_neighbors_mp,
                             s_log_neighbors_4_divided_pagerank_hp,
                             s_log_neighbors_4_divided_pagerank_mp,
                             s_log_neighbors_3_divided_pagerank_hp,
                             s_log_neighbors_3_divided_pagerank_mp,
                             s_log_neighbors_2_divided_pagerank_hp,
                             s_log_neighbors_2_divided_pagerank_mp,
                             s_neighbors_2_hp_divided_neighbors_2_mp,
                             s_neighbors_3_hp_divided_neighbors_3_mp,
                             s_neighbors_4_hp_divided_neighbors_4_mp,
                             s_log_truncatedpagerank_1_hp,
                             s_log_truncatedpagerank_2_hp,
                             s_log_truncatedpagerank_3_hp,
                             s_log_truncatedpagerank_4_hp,
                             s_log_truncatedpagerank_1_mp,
                             s_log_truncatedpagerank_2_mp,
                             s_log_truncatedpagerank_3_mp,
                             s_log_truncatedpagerank_4_mp,
                             s_log_truncatedpagerank_1_hp_divided_pagerank_hp,
                             s_log_truncatedpagerank_2_hp_divided_pagerank_hp,
                             s_log_truncatedpagerank_3_hp_divided_pagerank_hp,
                             s_log_truncatedpagerank_4_hp_divided_pagerank_hp,
                             s_log_truncatedpagerank_1_mp_divided_pagerank_mp,
                             s_log_truncatedpagerank_2_mp_divided_pagerank_mp,
                             s_log_truncatedpagerank_3_mp_divided_pagerank_mp,
                             s_log_truncatedpagerank_4_mp_divided_pagerank_mp,
                             s_truncatedpagerank_4_hp_divided_truncatedpagerank_3_hp,
                             s_truncatedpagerank_4_mp_divided_truncatedpagerank_3_mp,
                             s_truncatedpagerank_3_hp_divided_truncatedpagerank_2_hp,
                             s_truncatedpagerank_3_mp_divided_truncatedpagerank_2_mp,
                             s_truncatedpagerank_2_hp_divided_truncatedpagerank_1_hp,
                             s_truncatedpagerank_2_mp_divided_truncatedpagerank_1_mp,
                             s_log_min_truncatedpagerank_hp,
                             s_log_min_truncatedpagerank_mp,
                             s_log_max_truncatedpagerank_hp,
                             s_log_max_truncatedpagerank_mp,
                             s_log_avg_truncatedpagerank_hp,
                             s_log_avg_truncatedpagerank_mp,
                             #134s_truncatedpagerank_1_mp_divided_truncatedpagerank_1_hp,
                             #135s_truncatedpagerank_2_mp_divided_truncatedpagerank_2_hp,
                             #136s_truncatedpagerank_3_mp_divided_truncatedpagerank_3_hp,
                             #137s_truncatedpagerank_4_mp_divided_truncatedpagerank_4_hp,
                             #s_g,
                             #s_b,
                             s_HST_1,
                             s_HST_2,
                             s_HST_3,
                             s_HST_4,
                             s_HST_5,
                             s_HST_6,
                             s_HST_7,
                             s_HST_8,
                             s_HST_9,
                             s_HST_10,
                             s_HST_11,
                             s_HST_12,
                             s_HST_13,
                             s_HST_14,
                             s_HST_15,
                             s_HST_16,
                             s_HST_17,
                             s_HST_18,
                             s_HST_19,
                             s_HST_20,
                             s_HST_21,
                             s_HST_22,
                             s_HST_23,
                             s_HST_24,
                             s_HMG_25,
                             s_HMG_26,
                             s_HMG_27,
                             s_HMG_28,
                             s_HMG_29,
                             s_HMG_30,
                             s_HMG_31,
                             s_HMG_32,
                             s_HMG_33,
                             s_HMG_34,
                             s_HMG_35,
                             s_HMG_36,
                             s_HMG_37,
                             s_HMG_38,
                             s_HMG_39,
                             s_HMG_40,
                             s_HMG_41,
                             s_HMG_42,
                             s_HMG_43,
                             s_HMG_44,
                             s_HMG_45,
                             s_HMG_46,
                             s_HMG_47,
                             s_HMG_48,
                             s_AVG_49,
                             s_AVG_50,
                             s_AVG_51,
                             s_AVG_52,
                             s_AVG_53,
                             s_AVG_54,
                             s_AVG_55,
                             s_AVG_56,
                             s_AVG_57,
                             s_AVG_58,
                             s_AVG_59,
                             s_AVG_60,
                             s_AVG_61,
                             s_AVG_62,
                             s_AVG_63,
                             s_AVG_64,
                             s_AVG_65,
                             s_AVG_66,
                             s_AVG_67, 
                             s_AVG_68,
                             s_AVG_69,
                             s_AVG_70,
                             s_AVG_71,
                             s_AVG_72,
                             s_STD_73,
                             s_STD_74,
                             s_STD_75,
                             s_STD_76,
                             s_STD_77,
                             s_STD_78,
                             s_STD_79,
                             s_STD_80,
                             s_STD_81,
                             s_STD_82,
                             s_STD_83,
                             s_STD_84,
                             s_STD_85,
                             s_STD_86,
                             s_STD_87,
                             s_STD_88,
                             s_STD_89,
                             s_STD_90,
                             s_STD_91,
                             s_STD_92,
                             s_STD_93,
                             s_STD_94,
                             s_STD_95,
                             s_STD_96]).T


# In[947]:

############## RUS_multiple method ####################
nonspam_list,spam_list = split_spam_nonspam(label)

s = len(spam_list)
n = len(nonspam_list)
k = int(n/s)

new_sublist_list = []
for i in range(9):
    np.random.shuffle(nonspam_list)
    nonspam_subarray = np.array_split(nonspam_list,k)
    for j in range(len(nonspam_subarray)):
        new_list = nonspam_subarray[j].tolist() + spam_list
        new_sublist_list.append(new_list)


# In[907]:

############### GBDT classifier ###################
# return spamicity
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier 
from __future__ import division
def vote_Classifier_GBDT(Ds,training_features,labels,test_features):
    #classifier
    clf=GradientBoostingClassifier(learning_rate=0.005, n_estimators=60, max_depth=9,random_state=10,subsample=0.85)
    predictions=[0 for i in xrange(len(test_features))]
    for i in xrange(len(Ds)):
        #get feature vectors and labels of sub_training set
        training_subset=np.array([training_features[Ds[i][0]]])
        labels_subset=[labels[Ds[i][0]]]
        for j in xrange(len(Ds[i])-1):
            training_subset=np.concatenate((training_subset, np.array([training_features[Ds[i][j+1]]])), axis=0)
            labels_subset.append(labels[Ds[i][j+1]])
        #train & predict
        clf = clf.fit(training_subset, labels_subset)
        pred_sub=clf.predict(test_features)
        for j in xrange(len(pred_sub)):
            if pred_sub[j]==0:
                pred_sub[j]=-1
        predictions=[predictions[m]+pred_sub[m] for m in xrange(len(pred_sub))]
    #get the average score of all sub_classifiers
    predictions=[predictions[m]/len(Ds) for m in xrange(len(predictions))]
    return predictions


# In[908]:

############### Evaluation function (AUC) ##################
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
def auc1(true_label,score):
    fpr, tpr, thresholds = roc_curve(true_label, score, pos_label=1)
    return auc(fpr, tpr)
                


# In[945]:

################# Result ###################
spamicity_gbdt = vote_Classifier_GBDT(ds,training_features,label,testing_features)
print auc1(label_test,spamicity_gbdt)


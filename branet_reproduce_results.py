# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 17:54:48 2021

@author: jagtaps
"""


import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
from matplotlib import pyplot
import seaborn as sns
from sklearn.metrics import precision_recall_curve,recall_score,confusion_matrix
from sklearn.metrics import roc_curve,average_precision_score,auc
import scipy.io as sio
from sklearn.metrics import confusion_matrix
import ast
import networkx as nx
import statistics
from sklearn import metrics
import functions as f


nodes = pd.read_csv('panc_data//all_nodes.txt',delimiter = '\t', header = None)
mirna_fc_dat_overxp = pd.read_csv("mirna_fc_dat_overxp" )
mrna_fc_dat_underexp = pd.read_csv("mrna_fc_dat_underexp" )

ref_net = nx.read_edgelist("panc_data//ref_net")

mirna_list = mirna_fc_dat_overxp['mirna']
mrna_list =mrna_fc_dat_underexp['mrna']

branet = pd.read_csv('test_branet.emb',delimiter = ' ', index_col= 0)
branet = branet.sort_index()
branet_res = f.evalation(branet,ref_net,mirna_list,mrna_list)

multinet = pd.read_csv('test_multinet.emb', delimiter = ' ', index_col= 0, skiprows=1, header = None)
multinet = multinet.sort_index()
multinet.index = nodes[0].values
multinet_res = f.evalation(multinet,ref_net,mirna_list,mrna_list)

mashup = pd.read_csv('test_mashup.emb', delimiter = '\t', header = None)
mashup = mashup.transpose()
mashup.index = nodes[0].values
mashup_res = f.evalation(mashup,ref_net,mirna_list,mrna_list)


G = nx.read_edgelist("input_edgelist", edgetype=float, data=(('weight', float),))
deepnf = np.load('test_deepnf.pckl', allow_pickle=True)
deepnf = pd.DataFrame(data=deepnf)
deepnf.index = G.nodes 
deepnf_res = f.evalation(deepnf,ref_net,mirna_list,mrna_list)

branet_y_probs = f.y_probs(branet,ref_net,mirna_list,mrna_list)
multinet_y_probs = f.y_probs(multinet,ref_net,mirna_list,mrna_list)
mashup_res_y_probs = f.y_probs(mashup,ref_net,mirna_list,mrna_list)
deepnf_res_y_probs = f.y_probs(deepnf,ref_net,mirna_list,mrna_list)


ibrane_precision, ibrane_recall, thresholds = precision_recall_curve(branet_y_probs['y'], branet_y_probs['probs'])
mashup_precision, mashup_recall, thresholds = precision_recall_curve(mashup_res_y_probs['y'], mashup_res_y_probs['probs'])
multinet_precision, multinet_recall, thresholds = precision_recall_curve(multinet_y_probs['y'], multinet_y_probs['probs'])
deepf_precision, deepnf_recall, thresholds = precision_recall_curve(deepnf_res_y_probs['y'], deepnf_res_y_probs['probs'])


APiBrane = "%.3f" % auc(ibrane_recall, ibrane_precision)
APimultinet = "%.3f" % auc(multinet_recall, multinet_precision)
APimashup = "%.3f" % auc(mashup_recall, mashup_precision)
APdeepnf = "%.3f" % auc(deepnf_recall, deepf_precision)

print("AUPR:")
print("BRANet: " + str(APiBrane))
print("Multi-Net: " + str(APimultinet))
print("Mashup: " + str(APimashup))
print("deepNF: " + str(APdeepnf))


ind_list = [0, 9,99,199,399,599,799,999]

branet_res = branet_res[branet_res.index.isin(ind_list)]
mashup_res = mashup_res[mashup_res.index.isin(ind_list)]
multinet_res = multinet_res[multinet_res.index.isin(ind_list)]
deepnf_res = deepnf_res[deepnf_res.index.isin(ind_list)]










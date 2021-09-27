# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 19:16:16 2021

@author: jagtaps
"""

import pandas as pd
import numpy as np
import networkx as nx
import scipy.sparse as sparse
from scipy.sparse import csgraph
from theano import tensor as T
import logging
import theano
from theano import tensor as T

logger = logging.getLogger(__name__)
theano.config.exception_verbosity='high'

def PPMI_matrix(A, window, b):
    n = A.shape[0]
    vol = float(A.sum())
    L, d_rt = csgraph.laplacian(A, normed=True, return_diag=True)
    X = sparse.identity(n) - L
    S = np.zeros_like(X)
    X_power = sparse.identity(n)
    for i in range(window):
        logger.info("Compute matrix %d-th power", i+1)
        X_power = X_power.dot(X)
        S += X_power
    S *= vol / window / b
    D_rt_inv = sparse.diags(d_rt ** -1)
    M = D_rt_inv.dot(D_rt_inv.dot(S).T)
    m = T.matrix()
    f = theano.function([m], T.log(T.maximum(m, 1)))
    Y = f(M.todense().astype(theano.config.floatX))
    return sparse.csr_matrix(Y)


def embedd(X, dim):
    u, s, v = sparse.linalg.svds(X, dim, return_singular_vectors="u")
    return sparse.diags(np.sqrt(s)).dot(u.T).T


def evalation(emb,ref_net,mirna_list,mrna_list):
    
    ref_net = nx.Graph(ref_net.subgraph(np.array(emb.index)))
    node2vec = emb[emb.index.isin(ref_net.nodes)]
    
    a1 = nx.adjacency_matrix(ref_net)
    dat = pd.DataFrame(data = a1.todense(),columns= np.array(ref_net.nodes))
    dat = dat.set_index([np.array(ref_net.nodes)])
    dat = dat.sort_index()
    dat = dat.transpose().sort_index()
    ref_dat = dat.stack().reset_index()

    Mat = node2vec.values @ node2vec.values.transpose()
    MatNorm  = (Mat-Mat.min())/(Mat.max()-Mat.min())
    Mat = pd.DataFrame(MatNorm,columns= np.array(node2vec.index))
    Mat = Mat.set_index([np.array(node2vec.index)])
    Mat = Mat.sort_index()
    Mat = Mat.transpose().sort_index()
    node2vec_df = Mat.stack().reset_index()
    
    node2vec_df = node2vec_df[node2vec_df['level_0'].isin(np.array(mirna_list))]
    node2vec_df = node2vec_df[node2vec_df['level_1'].isin(np.array(mrna_list))]
    
    ref_dat = ref_dat[ref_dat['level_0'].isin(np.array(mirna_list))]
    ref_dat = ref_dat[ref_dat['level_1'].isin(np.array(mrna_list))]

    node2vec_datt = pd.DataFrame({'y': ref_dat[0], 'probs': node2vec_df[0]})
    node2vec_datt = node2vec_datt.sort_values('probs', ascending=False)
    
    dat = node2vec_datt[:1000]
    y = dat['y']
    s = []
    for i in range(1,len(y)+1):
        ss = (sum(y[:i].values)/i)
        s.append(ss)

    p = {'k': list(range(1,len(y)+1)), 'precision': s}
    datt = pd.DataFrame(p)
    return datt



def y_probs(emb,ref_net,mirna_list,mrna_list):
    
    ref_net = nx.Graph(ref_net.subgraph(np.array(emb.index)))
    node2vec = emb[emb.index.isin(ref_net.nodes)]
    
    a1 = nx.adjacency_matrix(ref_net)
    dat = pd.DataFrame(data = a1.todense(),columns= np.array(ref_net.nodes))
    dat = dat.set_index([np.array(ref_net.nodes)])
    dat = dat.sort_index()
    dat = dat.transpose().sort_index()
    ref_dat = dat.stack().reset_index()

    Mat = node2vec.values @ node2vec.values.transpose()
    MatNorm  = (Mat-Mat.min())/(Mat.max()-Mat.min())
    Mat = pd.DataFrame(MatNorm,columns= np.array(node2vec.index))
    Mat = Mat.set_index([np.array(node2vec.index)])
    Mat = Mat.sort_index()
    Mat = Mat.transpose().sort_index()
    node2vec_df = Mat.stack().reset_index()
    
    node2vec_df = node2vec_df[node2vec_df['level_0'].isin(np.array(mirna_list))]
    node2vec_df = node2vec_df[node2vec_df['level_1'].isin(np.array(mrna_list))]
    
    ref_dat = ref_dat[ref_dat['level_0'].isin(np.array(mirna_list))]
    ref_dat = ref_dat[ref_dat['level_1'].isin(np.array(mrna_list))]

    node2vec_datt = pd.DataFrame({'y': ref_dat[0], 'probs': node2vec_df[0]})
    node2vec_datt = node2vec_datt.sort_values('probs', ascending=False)
    
    
    return node2vec_datt



def evalln(emb,ref_net,mirna_list,mrna_list):
    
    ref_net = nx.Graph(ref_net.subgraph(np.array(emb.index)))
    a1 = nx.adjacency_matrix(ref_net)
    dat = pd.DataFrame(data = a1.todense(),columns= np.array(ref_net.nodes))
    dat = dat.set_index([np.array(ref_net.nodes)])
    dat = dat.sort_index()
    dat = dat.transpose().sort_index()
    ref_dat = dat.stack().reset_index()

    ref_dat = ref_dat[ref_dat['level_0'].isin(np.array(mirna_list))]
    ref_dat = ref_dat[ref_dat['level_1'].isin(np.array(mrna_list))]

    return ref_dat
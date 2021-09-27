# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 17:37:57 2021

@author: jagtaps
"""


import numpy as np
import networkx as nx
import pandas as pd
import requests
import gzip
import shutil
import mygene
import functions as f


mirna_dat_normal =  pd.read_csv("data//Pancancer_miRNA_Normal.csv" )
mirna_dat_tumor =  pd.read_csv("data//Pancancer_miRNA_Tumor.csv" )

mrna_dat_normal =  pd.read_csv("data//Pancancer_mRNA_Normal.csv" )
mrna_dat_tumor =  pd.read_csv("data//Pancancer_mRNA_Tumor.csv" )

ref_net = nx.read_edgelist("data//ref_net")

mirna_all = pd.concat([mirna_dat_normal, mirna_dat_tumor], axis=1)
mirna_all = mirna_all.loc[:,~mirna_all.columns.duplicated()]
mrna_all = pd.concat([mrna_dat_normal, mrna_dat_tumor], axis=1)
mrna_all = mrna_all.loc[:,~mrna_all.columns.duplicated()]

mirna_dat_N_mean = mirna_dat_normal.iloc[:,1:mirna_dat_normal.shape[1]].mean(axis=1)
mirna_dat_T_mean = mirna_dat_tumor.iloc[:,1:mirna_dat_tumor.shape[1]].mean(axis=1)

mrna_dat_N_mean = mrna_dat_normal.iloc[:,1:mrna_dat_normal.shape[1]].mean(axis=1)
mrna_dat_T_mean = mrna_dat_tumor.iloc[:,1:mrna_dat_tumor.shape[1]].mean(axis=1)


d = {'mirna' : np.array(mirna_dat_normal['miRNA'].values),'logFC' : np.log2(np.array(mirna_dat_T_mean)/np.array(mirna_dat_N_mean))}
mirna_fc_dat = pd.DataFrame(d)
mirna_fc_dat = mirna_fc_dat[mirna_fc_dat['logFC']>2]
mirna_fc_dat_overxp = mirna_fc_dat.replace([np.inf, -np.inf], np.nan).dropna(axis=0)


d = {'mrna' : np.array(mrna_dat_normal['mRNA'].values),'logFC' : np.log2(np.array(mrna_dat_T_mean)/np.array(mrna_dat_N_mean))}
mrna_fc_dat = pd.DataFrame(d)
mrna_fc_dat = mrna_fc_dat[mrna_fc_dat['logFC']<-2]
mrna_fc_dat_underexp = mrna_fc_dat.replace([np.inf, -np.inf], np.nan).dropna(axis=0)


mirna_overexp_dat = mirna_all.iloc[mirna_fc_dat_overxp.index]
mirna_overexp_datt = mirna_overexp_dat.iloc[:,1:mirna_overexp_dat.shape[1]]
mirna_overexp_datt.index = mirna_overexp_dat['miRNA'].values
mirna_coexp_dat = mirna_overexp_datt.transpose().corr().stack().reset_index()
mirna_coexp_dat = mirna_coexp_dat[mirna_coexp_dat[0]>0.8]
mirna_coexp_dat.to_csv('mirna.coexpnet', sep='\t',index= None)

mrna_overexp_dat = mrna_all.iloc[mrna_fc_dat_underexp.index]
mrna_overexp_datt = mrna_overexp_dat.iloc[:,1:mrna_overexp_dat.shape[1]]
mrna_overexp_datt.index = mrna_overexp_dat['mRNA'].values
mrna_coexp_dat = mrna_overexp_datt.transpose().corr().stack().reset_index()
mrna_coexp_dat = mrna_coexp_dat[mrna_coexp_dat[0]>0.8]
mrna_coexp_dat.to_csv('mrna.coexpnet', sep='\t',index= None)

mirna_mrna_coexp = pd.concat([mirna_coexp_dat,mrna_coexp_dat])
mirna_mrna_coexp_edgelist = mirna_mrna_coexp[['level_0','level_1']]


mirna_fc_dat_overxp.to_csv('mirna_fc_dat_overxp')
mrna_fc_dat_underexp.to_csv('mrna_fc_dat_underexp')


url = 'http://mirdb.org/mirdb/download/miRDB_v6.0_prediction_result.txt.gz'
r = requests.get(url)

with open('mirdb', 'wb') as f_in:
    f_in.write(r.content)

with gzip.open('mirdb', 'rb') as f_in:
    with open('mirdb.txt', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)


mirna_db =  pd.read_csv("miRDB_v6.0_prediction_result.txt", sep="\t", header=None)
mirna_db = mirna_db[mirna_db[0].isin(np.array(mirna_fc_dat_overxp['mirna']))]
mirna_db = mirna_db[mirna_db[2]>80]


mg = mygene.MyGeneInfo()
res = pd.DataFrame(mg.querymany(list(mirna_db[1].values), scopes='refseq'))
mirna_mrna_target = pd.DataFrame({'level_0': mirna_db[0].values, 'level_1' : res['symbol'].values})
mirna_mrna_target=mirna_mrna_target[mirna_mrna_target['level_1'].isin(np.array(mrna_fc_dat_underexp['mrna']))]


input_edgelist = pd.concat([mirna_mrna_coexp_edgelist,mirna_mrna_target])
input_edgelist.to_csv('input_edgelist', sep='\t',index= None, header = None)


e1 = nx.read_edgelist('input_edgelist')
a1 = nx.adjacency_matrix(e1)
m1 = f.PPMI_matrix(a1,2,1)
emb1 = f.embedd(m1,128)

dat = pd.DataFrame(data = emb1)
dat.index = e1.nodes
dat.to_csv('test_branet.emb', sep=' ')


mirna_list = mirna_fc_dat_overxp['mirna']
mrna_list =mrna_fc_dat_underexp['mrna']

top_mirna = mirna_fc_dat_overxp[:10]
mirna_list = top_mirna['mirna']

Mat = dat.values @ dat.values.transpose()
MatNorm  = (Mat-Mat.min())/(Mat.max()-Mat.min())
Mat = pd.DataFrame(MatNorm,columns= np.array(dat.index))
Mat = Mat.set_index([np.array(dat.index)])
Mat = Mat.sort_index()
Mat = Mat.transpose().sort_index()
Mat_df = Mat.stack().reset_index()


Mat_df = Mat_df.sort_values(0, ascending=False)
Mat_df.columns = ['source', 'target', 'weight']

G = nx.from_pandas_edgelist(Mat_df,edge_attr=True)
G.remove_edges_from(nx.selfloop_edges(G))
dat = nx.to_pandas_edgelist(G)
dat = dat.sort_values('weight', ascending=False)

Mat_df = dat[dat['source'].isin(np.array(mirna_list))]

top = 400
dat_top = Mat_df[:top]

top_mrna = dat_top['target'][dat_top['target'].isin(np.array(mrna_list))]

mrna_df = dat[dat['source'].isin(np.array(top_mrna))]
mrna_df = mrna_df[mrna_df['target'].isin(np.array(top_mrna))]
mrna_df = mrna_df[:100]

net = pd.concat([mrna_df,dat_top])
net.to_csv('branet_top500.txt', sep='\t',index=None)











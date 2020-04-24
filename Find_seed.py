"""
A simple implementation of Ultimatum Game in complex network
@date: 2020.4.20
@author: Tingyu Mo
"""

import os
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

def avg_degree_caculate( G):
    '''
    caculate average degree of graph
    '''
    degree_total = 0
    for x in range(len(G.degree())):
        degree_total = degree_total + G.degree(x)
    return degree_total/node_num

def nbr_weighted_check( G,n,max_weight):
    is_MaxWeight_exis = None
    for nbr in G.adj[n]:
        weight = G.edges[n, nbr]['weight']
        if weight ==  max_weight:
            is_MaxWeight_exis = nbr
            break
    return is_MaxWeight_exis

def network_weights_asign(G,max_weight,avg_degree):
    #边权重初始化
    for n in list(G.nodes()):
        nbrs = list(G.adj[n])
        for nbr in nbrs:
            G.edges[n, nbr]['weight'] = 0
    
    #检查双方是否存在紧密联系者
    for n in list(G.nodes()):
        nbrs = list(G.adj[n])
        for nbr in nbrs:
            isMaxWeightExisIn_N = nbr_weighted_check(G,n,max_weight)
            isMaxWeightExisIn_Nbr = nbr_weighted_check(G,nbr,max_weight)
            if (isMaxWeightExisIn_N == None) and (isMaxWeightExisIn_Nbr == None):
                G.edges[n, nbr]['weight'] = max_weight
            elif (isMaxWeightExisIn_N==nbr) and (isMaxWeightExisIn_Nbr == n):
                G.edges[n, nbr]['weight'] = max_weight
            elif (isMaxWeightExisIn_N != None) or (isMaxWeightExisIn_Nbr != None) :
                G.edges[n, nbr]['weight'] = (1-max_weight)/(avg_degree-1)
                
    # 打印输出
    # for n, nbrs in G.adjacency():
    #     for nbr, eattr in nbrs.items():
    #         data = eattr['weight']
    #         print('(%d, %d, %0.3f)' % (n,nbr,data))
    cnt = 0
    for n in list(G.nodes()):
        result = nbr_weighted_check(G,n,max_weight)
        if result == None:
            cnt += 1

    imitate_rate = 1.0*cnt/node_num
    # print("无亲密关系者率:",cnt/self.node_num)

    return G,imitate_rate
if __name__ == '__main__':
  
    avg_degree = 4
    node_num = 100
    max_weight = 0.55
    result_list = []
    save_path = r"./result/seed{}_.csv".format(np.random.rand())
    cnt = 0
    for seed in range(3000,3000):

        G = nx.random_graphs.random_regular_graph(avg_degree, node_num,seed) #seed == 1 ,98%
        G,imitate_rate =  network_weights_asign(G,max_weight,avg_degree)

        if imitate_rate <= 0.02:
            cnt +=1
            print("无亲密关系者率:", imitate_rate)
            print("随机种子:", seed) 
            re_avg_degree = avg_degree_caculate(G)
            print("平均连接度为: ", re_avg_degree ) 
            result_list .append([seed,imitate_rate,re_avg_degree])  
            result_pd = pd.DataFrame(data = result_list)
            result_pd.to_csv(save_path)

    print("共找到{}个随机种子适合网络建模！".format(cnt))
        

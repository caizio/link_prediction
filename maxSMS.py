import numpy as np
import networkx as nx

def _maxSMS(G,i,j):
    ds = 0
    for xx in G.neighbors(i):
        for yy in G.neighbors(j):
            d = 0
            if xx != yy and G.has_edge(xx,yy):
                common_neighbors_i_yy = len(list(nx.common_neighbors(G, i, yy)))
                common_neighbors_j_xx = len(list(nx.common_neighbors(G, j, xx)))
                degree_xx = G.degree(xx)
                degree_yy = G.degree(yy)
                d = common_neighbors_j_xx * common_neighbors_i_yy / degree_xx / degree_yy
            ds = max(d,ds)
    return ds

def maxSMS(train):
    G = nx.from_numpy_array(train)
    n = train.shape[0]
    result = np.zeros((n,n))
    for i in range(0,n):
        for j in range(i+1,n):
            if train[i,j] == 0:
                d = _maxSMS(G,i,j)
            result[i,j] = d
            result[j,i] = d
    return result

def _SMS(G,i,j):
    ds = 0
    for xx in G.neighbors(i):
        for yy in G.neighbors(j):
            d = 0
            if xx != yy and G.has_edge(xx,yy):
                common_neighbors_i_yy = len(list(nx.common_neighbors(G, i, yy)))
                common_neighbors_j_xx = len(list(nx.common_neighbors(G, j, xx)))
                degree_xx = G.degree(xx)
                degree_yy = G.degree(yy)
                d = common_neighbors_j_xx * common_neighbors_i_yy / degree_xx / degree_yy
            ds += d
    return ds


def SMS(train):
    G = nx.from_numpy_array(train)
    n = train.shape[0]
    result = np.zeros((n,n))
    for i in range(0,n):
        for j in range(i+1,n):
            if train[i,j] == 0:
                d = _SMS(G,i,j)
            result[i,j] = d
            result[j,i] = d
    return result
    

    
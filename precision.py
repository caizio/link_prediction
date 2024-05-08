import numpy as np

# testPair表示测试集边的集合,similar是相似性矩阵,需要检查代码是否有误
def acc_precision(trainMatrix,testPair,similar,top_L = 500.0):
    edges_score = []
    n = similar.shape[0]
    for i in range(n):
        for j in range(i+1,n):
            if similar[i,j] != 0 and trainMatrix[i,j] == 0:
                edge = (i,j)
                edge_score = [edge,similar[i,j]]
                edges_score.append(edge_score)
    sorted_edges = sorted(edges_score, key=lambda x: x[1], reverse=True)
    
    k = 0
    for i in range(top_L):
        e = list(sorted_edges[i][0])
        if e in testPair:
            k += 1
    # result = k / top_L    
    return k / top_L
    
def count_testPair_score(trainMatrix,testPair,similar):
    edges_score = {}
    n = similar.shape[0]
    for i in range(n):
        for j in range(i+1,n):
            if similar[i,j] != 0 and trainMatrix[i,j] == 0 and i != j:
                edge = (i,j)
                edges_score[edge] = similar[i,j]
    
    score = []
    for testP in testPair:
        testpp = tuple(testP)
        if(edges_score.get(testpp) != None):
            score.append(edges_score.get(testpp))
        else:
            score.append(0)  
    return score        
        
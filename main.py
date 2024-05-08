# 计算auc
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

import dataDeal as dd
import linkPrediciton as lp
import AUC as acc
import precision as precision


n = 1

trainPaths = [
    "./dataset/train.txt",
    "./dataset/USAirTrain_0.txt",
    "./dataset/C. elegansTrain_90%0.txt",
    "./dataset/S. pombeTrain_90%0.txt",
    "./dataset/S. pombeTrain_50%0.txt"
]

testPaths = [
    "./dataset/test.txt",
    "./dataset/USAirTest_0.txt",
    "./dataset/C. elegansTest_10%0.txt",
    "./dataset/S. pombeTest_10%0.txt",
    "./dataset/S. pombeTest_50%0.txt"
]

# 读取数据,train、test以边的形式储存，trainMatrix、testMatrix以矩阵的形式储存数据
trainPath = trainPaths[4]
testPath = testPaths[4]
trainPair,maxNodeNum1 = dd.readEdge(trainPath)
testPair,maxNodeNum2 = dd.readEdge(testPath)
maxNodeNum = max(maxNodeNum1,maxNodeNum2)
trainMatrix = dd.edgeToMatrix(trainPair,maxNodeNum)
testMatrix = dd.edgeToMatrix(testPair,maxNodeNum)
G = nx.from_numpy_array(trainMatrix)

for i in range(n):

    # 计算相似性矩阵并计算auc
    similar = lp.CN(trainMatrix)
    auc = acc.Calculation_AUC(trainMatrix,testMatrix,similar,maxNodeNum+1)
    print("CN")
    print("AUC:%f"%auc)   
    p = precision.acc_precision(trainMatrix,testPair,similar,500)
    print("P:%f" % p)

    similar = lp.A3(trainMatrix)
    auc = acc.Calculation_AUC(trainMatrix,testMatrix,similar,maxNodeNum+1)
    print("A3")
    print("AUC:%f"%auc)   
    p = precision.acc_precision(trainMatrix,testPair,similar,500)
    print("P:%f" % p)

    similar = lp.Katz(trainMatrix)
    auc = acc.Calculation_AUC(trainMatrix,testMatrix,similar,maxNodeNum+1)
    print("Katz")
    print("AUC:%f"%auc)
    p = precision.acc_precision(trainMatrix,testPair,similar,500)
    print("P:%f" % p)

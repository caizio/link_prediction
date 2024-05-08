# 数据处理，划分数据集（包括训练集和测试集）

import pandas as pd
import random
import numpy as np

# 读取原始数据,并处理后写出到edge.txt和map.txt中
def readDataAndWriteData(path:str):
    df = pd.read_excel(path,index_col=False)
    label = list(df)[1:]
    n = len(label)
    data = []
    for i in range(n):
        for j in range(i+1,n+1):
            if j == 1:
                continue
            if df.loc[i][j] != 0:
                temp = []
                temp.append(i)
                temp.append(j-1)
                temp.append(df.loc[i][j])
                data.append(temp)
    with open("./dataset/map.txt",'w') as f:
        for i in range(len(label)):
            f.write(str(i) + "," + label[i] + "\n")
    with open("./dataset/edge.txt",'w') as f:
        for edge in data:
            f.write(str(edge[0]) + "," + str(edge[1]) + "," + str(edge[2]) + "\n") 
    with open("./dataset/edge2.txt",'w') as f:
        for edge in data:
            f.write(str(edge[0]) + "," + str(edge[1]) + "\n") 
    # print(label)
    # print(data)
# 读取边
def readEdge(path:str):
    maxNode = 0
    edge = []
    with open(path,"r") as f:
        lines = f.readlines()
        for line in lines:
            temp = line.replace("\n","").split(",")
            temp2 = [int(d) for d in temp]
            if temp2[0] > temp2[1]:
                index = temp2[1]
                temp2[1] = temp2[0]
                temp2[0] = index
            if temp[0] == temp[1]:
                continue
            maxNode = max(maxNode,temp2[0],temp2[1])
            edge.append(temp2)
    # print(edge)
    return edge,maxNode
# 读取名称
def readName(path:str):
    name = []
    with open(path,"r") as f:
        lines = f.readlines()
        for line in lines:
            temp = line.replace("\n","").split(",")
            name.append(temp[1])
    return name
# 写出边
def writeEdge(path:str,data:list):
    with open(path,'w') as f:
        for edge in data:
            if len(edge) == 2:
                f.write(str(edge[0]) + "," + str(edge[1])+ "\n")  
            if len(edge) == 3:
                f.write(str(edge[0]) + "," + str(edge[1]) + "," + str(edge[2]) + "\n") 

# 数据集划分(随机划分),n为划分的比例，0.9代表9:1
def divide(edge:list,n = 0.9):
    edge_size = len(edge)
    train_size =  int(edge_size * n)
    train = []
    test = []
    select_number = edge_size - 1
    for i in range(train_size):
        select = random.randint(0,select_number)
        train.append(edge[select])
        edge.pop(select)
        select_number -= 1
    test = edge
    return train,test

# 将边的数据类型转化成矩阵(numpy)的形式
def edgeToMatrix(datas:list,maxNodeNum:int):
    matrix = np.zeros([maxNodeNum+1,maxNodeNum+1])
    for data in datas:
        i = data[0]
        j = data[1]
        if len(data) == 3:
            matrix[i,j] = data[2]
            matrix[j,i] = data[2]
        if len(data) == 2:
            matrix[i,j] = 1
            matrix[j,i] = 1
    return matrix  
# 将边的数据类型转化成矩阵的形式
def edgeToMatrix2(datas:list,maxNodeNum:int):
    matrix = [[j for j in range(maxNodeNum+1)] for i in range(maxNodeNum+1)]
    for data in datas:
        i = data[0]
        j = data[1]
        if len(data) == 3:
            matrix[i][j] = data[2]
            matrix[j][i] = data[2]
        if len(data) == 2:
            matrix[i][j] = 1
            matrix[j][i] = 1
    return matrix  
# 写出预测的矩阵
def writeSimilar(path:str,names:list,similar,edgeMatrix):
    n = len(names)
    with open(path,'w') as f:
        f.write(",")
        for name in names:
            f.write(name) 
            f.write(",")
        f.write("\n")
        for i in range(n):
            f.write(names[i])
            f.write(",")
            for j in range(n):
                # 如果是已有的边，输出0
                if (edgeMatrix[i,j] != 0) or (edgeMatrix[j,i] != 0):
                    f.write("0")
                    f.write(",")      
                    continue             
                if i < j:
                    f.write(str(similar[i,j]))
                    f.write(",")
                elif i == j:
                    f.write("0")
                    f.write(",")
                else:
                    f.write(str(similar[j,i]))
                    f.write(",")    
            f.write("\n")                
# 将矩阵转化为边的形式
def matrixToEdge(similar):
    n = similar.shape[0]
    edge = []
    for i in range(n):
        for j in range(i+1,n):
            if similar[i,j] !=0:
                temp = [i,j,similar[i,j]]
                edge.append(temp)
    return edge
# 写出得分最高的100条边,参数为0时,写出所有的边
def writeEdgeRankByScore(path:str,pedge:list,namses:list,edgeMatrix:list,n = 0):
    if n >= len(pedge):
        print("error::writeEdgeRankByScore::没有这么多条边可以写出")
    if n == 0:
        n = len(pedge)
    with open(path,'w') as f:
        for i in range(n):
            # 如果原始数据存在，无需预测，则删除
            if edgeMatrix[pedge[i][0]][pedge[i][1]] != 0:
                continue
            name1 = str(namses[pedge[i][0]])
            name2 = str(namses[pedge[i][1]])
            socre = str(pedge[i][2])
            f.write(name1 + "," + name2 + "," + socre + ",\n")
            
if __name__ == "__main__":
    path = "./dataset/数据.xlsx"
    readDataAndWriteData(path)
    edge,maxNode = readEdge("./dataset/edge.txt")
    train,test = divide(edge)
    writeEdge("./dataset/train.txt",train)
    writeEdge("./dataset/test.txt",test)
    print("ok")
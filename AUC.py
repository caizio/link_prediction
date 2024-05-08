import numpy as np
import random

# 参考代码里求auc的方法
def Calculation_AUC(MatrixAdjacency_Train,MatrixAdjacency_Test,Matrix_similarity,MaxNodeNum):
    AUCnum = 672400
    Matrix_similarity = np.triu(Matrix_similarity - Matrix_similarity * MatrixAdjacency_Train)
    Matrix_NoExist = np.ones(MaxNodeNum) - MatrixAdjacency_Train - MatrixAdjacency_Test - np.eye(MaxNodeNum)
    Test = np.triu(MatrixAdjacency_Test)
    NoExist = np.triu(Matrix_NoExist)
    Test_num = len(np.argwhere(Test == 1))
    NoExist_num = len(np.argwhere(NoExist == 1))
    Test_rd = [int(x) for index,x in enumerate((Test_num * np.random.rand(1,AUCnum))[0])]
    NoExist_rd = [int(x) for index,x in enumerate((NoExist_num * np.random.rand(1,AUCnum))[0])]
    TestPre= Matrix_similarity * Test
    NoExistPre = Matrix_similarity * NoExist
    TestIndex = np.argwhere(Test == 1)
    Test_Data = np.array([TestPre[x[0],x[1]] for index,x in enumerate(TestIndex)]).T
    NoExistIndex = np.argwhere(NoExist == 1)
    NoExist_Data = np.array([NoExistPre[x[0],x[1]] for index,x in enumerate(NoExistIndex)]).T
    Test_rd = np.array([Test_Data[x] for index,x in enumerate(Test_rd)])
    NoExist_rd = np.array([NoExist_Data[x] for index,x in enumerate(NoExist_rd)])
    n1,n2 = 0,0
    for num in range(AUCnum):
        if Test_rd[num] > NoExist_rd[num]:
            n1 += 1
        elif Test_rd[num] == NoExist_rd[num]:
            n2 += 0.5
        else:
            n1 += 0
    auc = float(n1+n2)/AUCnum
    return auc

# 复现的计算AUC的方法,两个方法结果是一样的
def acc_AUC(train,test,similar,AUCnum = 672400):
    n = len(train)
    if n == 0 or test.shape[0] == 0:
        print("error::acc_AUC::train or test is wrong")
        return -1
    test_edge = []
    non_edge = []
    for i in range(n):
        for j in range(i + 1,n):
            if test[i,j] != 0:
                test_edge.append([i,j])
            if train[i,j] == 0 and test[i,j] == 0:
                non_edge.append([i,j])
    sum = 0.0
    test_size = len(test_edge)
    non_size = len(non_edge)
    # 每次从测试集和要预测的集合（全集-训练集-测试集）中挑选一条边，比较得分大小，若测试集分大+1分，相等则+0.5分
    random.seed()
    for i in range(AUCnum):
        ranint1 = random.randint(0,test_size-1)
        ranint2 = random.randint(0,non_size-1)
        testScore = similar[test_edge[ranint1][0],test_edge[ranint1][1]]
        nonScore = similar[non_edge[ranint2][0],non_edge[ranint2][1]]
        if  testScore >  nonScore:
            sum += 1.0
        elif testScore == nonScore:
            sum += 0.5
        else:
            pass
    auc = sum / AUCnum
    return auc
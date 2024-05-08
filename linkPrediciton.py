import numpy as np

def CN(train):
    similarity = np.dot(train,train)
    return similarity

def _CN2(train,x,y):
    n = train.shape[0]
    set_x = set()
    set_y = set()
    for i in range(n):
        if train[x][i] != 0:
            set_x.add(i)
        if train[y][i] != 0:
            set_y.add(i)
    cn = set_x.intersection(set_y)
    cn_number = len(cn)
    return cn_number
def CN2(train):
    n = len(train)
    similar = np.zeros([n,n])
    # similar = [[0 for col in range(n)] for row in range(n)]
    for i in range(n):
        for j in range(i+1,n):
            if train[i,j] == 0:
                si = _CN2(train,i,j)
                similar[i,j] = si
                similar[j,i] = si
    return similar

def _JC(train,x,y):
    n = train.shape[0]
    set_x = set()
    set_y = set()
    for i in range(n):
        if train[x][i] != 0:
            set_x.add(i)
        if train[y][i] != 0:
            set_y.add(i)
    cn = set_x.intersection(set_y)
    un = set_x.union(set_y)
    cn_number = len(cn)
    un_number = len(un)
    if un_number == 0:
        return 0
    return cn_number / un_number
def JC(train):
    n = len(train)
    similar = np.zeros([n,n])
    # similar = [[0 for col in range(n)] for row in range(n)]
    for i in range(n):
        for j in range(i+1,n):
            if train[i,j] == 0:
                si = _JC(train,i,j)
                similar[i,j] = si
                similar[j,i] = si
    return similar    
def Katz(train):
    Parameter = 0.01
    Matrix_EYE = np.eye(train.shape[0])
    Temp = Matrix_EYE - train * Parameter
    similarity = np.linalg.inv(Temp)
    similarity = similarity - Matrix_EYE
    return similarity

def A3(train):
    return train @ train @ train


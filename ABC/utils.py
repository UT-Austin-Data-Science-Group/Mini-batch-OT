# Author: Kimia Nadjahi
# Some parts of this code are taken from https://github.com/skolouri/swgmm
from sklearn.cluster import KMeans
import numpy as np
import ot
# import HilbertCode_caller
# import swapsweep

def mOT(x,y,k,m):
    n= x.shape[0]
    if(k  < int(n/m)):
        inds1= np.split(np.random.permutation(n)[:int(n/m)*m], int(n/m))
        inds2 = np.split(np.random.permutation(n)[:int(n/m)*m], int(n/m))
        inds1 = list(np.array(inds1)[np.random.choice(len(inds1), k, replace=False)])
        inds2 = list(np.array(inds2)[np.random.choice(len(inds2), k, replace=False)])
    else:
        num_permute=int(k/int(n/m))+1
        inds1=[]
        inds2=[]
        for _ in range(num_permute):
            inds1_p = np.split(np.random.permutation(n), int(n / m))
            inds2_p = np.split(np.random.permutation(n), int(n / m))
            inds1+=inds1_p
            inds2+=inds2_p
        inds1 = list(np.array(inds1)[np.random.choice(len(inds1), k, replace=False)])
        inds2 = list(np.array(inds2)[np.random.choice(len(inds2), k, replace=False)])
    # C = ot.dist(x, y)
    cost=0
    for i in range(k):
        for j in range(k):
            M= ot.dist(x[inds1[i]], y[inds2[j]])
            C=ot.emd([], [], M)
            cost+= np.sum(C*M)
    return cost/(k**2)

def BoMbOT(x,y,k,m):
    n = x.shape[0]
    if (k  < int(n/m)):
        inds1 = np.split(np.random.permutation(n)[:int(n/m)*m], int(n / m))
        inds2 = np.split(np.random.permutation(n)[:int(n/m)*m], int(n / m))
        inds1 = list(np.array(inds1)[np.random.choice(len(inds1), k, replace=False)])
        inds2 = list(np.array(inds2)[np.random.choice(len(inds2), k, replace=False)])
    else:
        num_permute = int(k/int(n/m))+1
        inds1 = []
        inds2 = []
        for _ in range(num_permute):
            inds1_p = np.split(np.random.permutation(n), int(n / m))
            inds2_p = np.split(np.random.permutation(n), int(n / m))
            inds1 += inds1_p
            inds2 += inds2_p
        inds1 = list(np.array(inds1)[np.random.choice(len(inds1), k, replace=False)])
        inds2 = list(np.array(inds2)[np.random.choice(len(inds2), k, replace=False)])
    # C = ot.dist(x, y)
    big_C= np.zeros((k,k))
    for i in range(k):
        for j in range(k):
            M= ot.dist(x[inds1[i]], y[inds2[j]])
            C=ot.emd([], [], M)
            big_C[i,j]=np.sum(C*M)
    pi=ot.emd([],[],big_C)
    return np.sum(pi*big_C)



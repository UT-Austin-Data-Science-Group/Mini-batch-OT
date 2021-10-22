import numpy as np
import ot
import random
import tqdm
def transform_mOT(src,target,src_label,origin,k,m,iter=100):
    np.random.seed(1)
    random.seed(1)
    ot_transf = np.zeros_like(src)
    n = src.shape[0]
    for _ in tqdm.tqdm(range(iter)):
        s = np.copy(src).reshape(-1, 3).astype('long')
        t = np.array(target).reshape(-1, 3).astype('long')
        inds1 = np.random.choice(n, k * m,replace=False).reshape(k, m).tolist()
        inds2 = np.random.choice(n, k * m,replace=False).reshape(k, m).tolist()
        for mi in range(k):
            for mj in range(k):
                indms=inds1[mi]
                indmt=inds2[mj]
                ms=s[indms]
                mt=t[indmt]
                M = ot.dist(ms, mt)
                plan = ot.emd([], [], M, numItermax=500000)
                ot_transf[indms]+= 1./(k**2) *m*plan.dot(t[indmt])
        # ot_transf=ot_transf/(k**2)*255
    img_ot_transf = ot_transf[src_label].reshape(origin.shape)
    img_ot_transf = img_ot_transf/np.max(img_ot_transf)*255
    img_ot_transf =img_ot_transf.astype('uint8')
    return ot_transf,img_ot_transf
def transform_mUOT(src,target,src_label,origin,k,m,reg,tau,iter=100):
    np.random.seed(1)
    random.seed(1)
    ot_transf = np.zeros_like(src)
    n=src.shape[0]
    for _ in tqdm.tqdm(range(iter)):
        s = np.copy(src).reshape(-1, 3).astype('long')
        t = np.array(target).reshape(-1, 3).astype('long')
        inds1 = np.random.choice(n, k * m, replace=False).reshape(k, m).tolist()
        inds2 = np.random.choice(n, k * m, replace=False).reshape(k, m).tolist()
        for mi in range(k):
            for mj in range(k):
                indms = inds1[mi]
                indmt = inds2[mj]
                ms = s[indms]
                mt = t[indmt]
                M = ot.dist(ms, mt)
                plan = ot.unbalanced.sinkhorn_knopp_unbalanced(np.ones(m)/m,np.ones(m)/m,M,reg=reg,reg_m=tau)
                ot_transf[indms] += 1./(k**2) *m*plan.dot(t[indmt])
        # ot_transf=ot_transf/(k**2)*255
    img_ot_transf = ot_transf[src_label].reshape(origin.shape)
    img_ot_transf = img_ot_transf / np.max(img_ot_transf) * 255
    img_ot_transf = img_ot_transf.astype('uint8')
    return ot_transf,img_ot_transf

def transform_mPOT(src,target,src_label,origin,k,m,mass,iter=100):
    np.random.seed(1)
    random.seed(1)
    ot_transf = np.zeros_like(src)
    n = src.shape[0]
    for _ in tqdm.tqdm(range(iter)):
        s = np.copy(src).reshape(-1, 3).astype('long')
        t = np.array(target).reshape(-1, 3).astype('long')
        inds1 = np.random.choice(n, k * m, replace=False).reshape(k, m).tolist()
        inds2 = np.random.choice(n, k * m, replace=False).reshape(k, m).tolist()
        for mi in range(k):
            for mj in range(k):
                indms = inds1[mi]
                indmt = inds2[mj]
                ms = s[indms]
                mt = t[indmt]
                M = ot.dist(ms, mt)
                plan = ot.partial.partial_wasserstein(np.ones(m)/m, np.ones(m)/m, M,m=mass)
                ot_transf[indms] += 1./(k**2) *m*plan.dot(t[indmt])
        # ot_transf=ot_transf/(k**2)*255
    img_ot_transf = ot_transf[src_label].reshape(origin.shape)
    img_ot_transf = img_ot_transf / np.max(img_ot_transf) * 255
    img_ot_transf = img_ot_transf.astype('uint8')
    return ot_transf, img_ot_transf
def transform_BombPOT(src,target,src_label,origin,k,m,mass,iter=100):
    np.random.seed(1)
    random.seed(1)
    ot_transf = np.zeros_like(src)
    n = src.shape[0]
    for _ in tqdm.tqdm(range(iter)):
        s = np.copy(src).reshape(-1, 3).astype('long')
        t = np.array(target).reshape(-1, 3).astype('long')
        inds1 = np.random.choice(n, k * m, replace=False).reshape(k, m).tolist()
        inds2 = np.random.choice(n, k * m, replace=False).reshape(k, m).tolist()
        big_C = np.zeros((k, k))
        for mi in range(k):
            for mj in range(k):
                indms=inds1[mi]
                indmt=inds2[mj]
                ms=s[indms]
                mt=t[indmt]
                M = ot.dist(ms, mt)
                plan = ot.partial.partial_wasserstein(np.ones(m)/m, np.ones(m)/m, M,m=mass)
                big_C[mi][mj] = np.sum(plan*M)
        pi = ot.emd([],[],big_C,numItermax=500000)
        for mi in range(k):
            for mj in range(k):
                indms=inds1[mi]
                indmt=inds2[mj]
                ms=s[indms]
                mt=t[indmt]
                M = ot.dist(ms, mt)
                plan = ot.partial.partial_wasserstein(np.ones(m)/m, np.ones(m)/m, M,m=mass)
                ot_transf[indms] += m*pi[mi][mj]*plan.dot(t[indmt])
    # ot_transf=ot_transf/(k**2)*255
    img_ot_transf = ot_transf[src_label].reshape(origin.shape)
    img_ot_transf = img_ot_transf/np.max(img_ot_transf)*255
    img_ot_transf =img_ot_transf.astype('uint8')
    return ot_transf,img_ot_transf
def transform_BombUOT(src,target,src_label,origin,k,m,reg,tau,iter=100):
    np.random.seed(1)
    random.seed(1)
    ot_transf = np.zeros_like(src)
    n = src.shape[0]
    for _ in tqdm.tqdm(range(iter)):
        s = np.copy(src).reshape(-1, 3).astype('long')
        t = np.array(target).reshape(-1, 3).astype('long')
        inds1 = np.random.choice(n, k * m, replace=False).reshape(k, m).tolist()
        inds2 = np.random.choice(n, k * m, replace=False).reshape(k, m).tolist()
        big_C = np.zeros((k, k))
        for mi in range(k):
            for mj in range(k):
                indms = inds1[mi]
                indmt = inds2[mj]
                ms = s[indms]
                mt = t[indmt]
                M = ot.dist(ms, mt)
                plan = ot.unbalanced.sinkhorn_knopp_unbalanced(np.ones(m)/m,np.ones(m)/m,M,reg=reg,reg_m=tau)
                big_C[mi][mj] = np.sum(plan * M)
        pi = ot.emd([], [], big_C, numItermax=500000)
        for mi in range(k):
            for mj in range(k):
                indms = inds1[mi]
                indmt = inds2[mj]
                ms = s[indms]
                mt = t[indmt]
                M = ot.dist(ms, mt)
                plan = ot.unbalanced.sinkhorn_knopp_unbalanced(np.ones(m)/m,np.ones(m)/m,M,reg=reg,reg_m=tau)
                ot_transf[indms] += m*pi[mi][mj] * plan.dot(t[indmt])
    # ot_transf=ot_transf/(k**2)*255
    img_ot_transf = ot_transf[src_label].reshape(origin.shape)
    img_ot_transf = img_ot_transf / np.max(img_ot_transf) * 255
    img_ot_transf = img_ot_transf.astype('uint8')
    return img_ot_transf
def transform_BombOT(src,target,src_label,origin,k,m,iter=100):
    np.random.seed(1)
    random.seed(1)
    ot_transf = np.zeros_like(src)
    n = src.shape[0]
    for _ in tqdm.tqdm(range(iter)):
        s = np.copy(src).reshape(-1, 3).astype('long')
        t = np.array(target).reshape(-1, 3).astype('long')
        inds1 = np.random.choice(n, k * m, replace=False).reshape(k, m).tolist()
        inds2 = np.random.choice(n, k * m, replace=False).reshape(k, m).tolist()
        big_C = np.zeros((k, k))
        for mi in range(k):
            for mj in range(k):
                indms = inds1[mi]
                indmt = inds2[mj]
                ms = s[indms]
                mt = t[indmt]
                M = ot.dist(ms, mt)
                plan = ot.emd([], [], M, numItermax=500000)
                big_C[mi][mj] = np.sum(plan * M)
        pi = ot.emd([], [], big_C, numItermax=500000)
        for mi in range(k):
            for mj in range(k):
                if(pi[mi,mj]==0):
                    continue
                indms = inds1[mi]
                indmt = inds2[mj]
                ms = s[indms]
                mt = t[indmt]
                M = ot.dist(ms, mt)
                plan = ot.emd([], [], M,numItermax=500000)
                ot_transf[indms] += m*pi[mi][mj] * plan.dot(t[indmt])
    # ot_transf=ot_transf/(k**2)*255
    img_ot_transf = ot_transf[src_label].reshape(origin.shape)
    img_ot_transf = img_ot_transf / np.max(img_ot_transf) * 255
    img_ot_transf = img_ot_transf.astype('uint8')
    return ot_transf,img_ot_transf



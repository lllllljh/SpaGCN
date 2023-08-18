import scanpy as sc
import pandas as pd
import numpy as np
import numba
import scipy
import os
from anndata import AnnData, read_csv, read_text, read_mtx
from scipy.sparse import issparse
import random
import torch
from SpaGCN import SpaGCN


# L2 distance
@numba.njit("f4(f4[:], f4[:])")
def euclid_dist(t1, t2):
    sum = 0
    for i in range(t1.shape[0]):
        sum += (t1[i] - t2[i]) ** 2
    return np.sqrt(sum)


# distance matrices
@numba.njit("f4[:,:](f4[:,:])", parallel=True, nogil=True)
def pairwise_distance(X):
    n = X.shape[0]
    adj = np.empty((n, n), dtype=np.float32)
    for i in numba.prange(n):
        for j in numba.prange(n):
            adj[i][j] = euclid_dist(X[i], X[j])
    return adj


def extract_color(x_pixel=None, y_pixel=None, image=None, beta=49):
    # beta to control the range of neighbourhood when calculate grey vale for one spot
    # range
    beta_half = round(beta / 2)
    g = []
    for i in range(len(x_pixel)):
        max_x = image.shape[0]
        max_y = image.shape[1]
        nbs = image[max(0, x_pixel[i] - beta_half):min(max_x, x_pixel[i] + beta_half + 1),
              max(0, y_pixel[i] - beta_half):min(max_y, y_pixel[i] + beta_half + 1)]
        g.append(np.mean(np.mean(nbs, axis=0), axis=0))
    # RGB normalize
    c0, c1, c2 = [], [], []
    for i in g:
        c0.append(i[0])
        c1.append(i[1])
        c2.append(i[2])
    c0 = np.array(c0)
    c1 = np.array(c1)
    c2 = np.array(c2)
    c3 = (c0 * np.var(c0) + c1 * np.var(c1) + c2 * np.var(c2)) / (np.var(c0) + np.var(c1) + np.var(c2))
    return c3


def calculate_adj_matrix(x, y, x_pixel=None, y_pixel=None, image=None, beta=49, alpha=1, histology=True):
    # x,y,x_pixel, y_pixel are lists
    if histology:
        assert (x_pixel is not None) & (x_pixel is not None) & (image is not None)
        assert (len(x) == len(x_pixel)) & (len(y) == len(y_pixel))

        # beta to control the range of neighbourhood when calculate grey vale for one spot
        # alpha to control the color scale
        beta_half = round(beta / 2)
        g = []
        for i in range(len(x_pixel)):
            max_x = image.shape[0]
            max_y = image.shape[1]
            nbs = image[max(0, x_pixel[i] - beta_half):min(max_x, x_pixel[i] + beta_half + 1),
                  max(0, y_pixel[i] - beta_half):min(max_y, y_pixel[i] + beta_half + 1)]
            g.append(np.mean(np.mean(nbs, axis=0), axis=0))
        c0, c1, c2 = [], [], []
        for i in g:
            c0.append(i[0])
            c1.append(i[1])
            c2.append(i[2])
        c0 = np.array(c0)
        c1 = np.array(c1)
        c2 = np.array(c2)
        print("Var of c0,c1,c2 = ", np.var(c0), np.var(c1), np.var(c2))
        c3 = (c0 * np.var(c0) + c1 * np.var(c1) + c2 * np.var(c2)) / (np.var(c0) + np.var(c1) + np.var(c2))
        # Gaussian
        c4 = (c3 - np.mean(c3)) / np.std(c3)
        z_scale = np.max([np.std(x), np.std(y)]) * alpha
        z = c4 * z_scale
        z = z.tolist()
        print("Var of x,y,z = ", np.var(x), np.var(y), np.var(z))
        X = np.array([x, y, z]).T.astype(np.float32)
    else:
        print("Calculating adj matrix using xy only...")
        X = np.array([x, y]).T.astype(np.float32)
    return pairwise_distance(X)


def prefilter_genes(adata, min_counts=None, max_counts=None, min_cells=10, max_cells=None):
    if min_cells is None and min_counts is None and max_cells is None and max_counts is None:
        raise ValueError('Provide one of min_counts, min_genes, max_counts or max_genes.')
    id_tmp = np.asarray([True] * adata.shape[1], dtype=bool)
    id_tmp = np.logical_and(id_tmp,
                            sc.pp.filter_genes(adata.X, min_cells=min_cells)[0]) if min_cells is not None else id_tmp
    id_tmp = np.logical_and(id_tmp,
                            sc.pp.filter_genes(adata.X, max_cells=max_cells)[0]) if max_cells is not None else id_tmp
    id_tmp = np.logical_and(id_tmp,
                            sc.pp.filter_genes(adata.X, min_counts=min_counts)[0]) if min_counts is not None else id_tmp
    id_tmp = np.logical_and(id_tmp,
                            sc.pp.filter_genes(adata.X, max_counts=max_counts)[0]) if max_counts is not None else id_tmp
    adata._inplace_subset_var(id_tmp)


def prefilter_specialgenes(adata, Gene1Pattern="ERCC", Gene2Pattern="MT-"):
    id_tmp1 = np.asarray([not str(name).startswith(Gene1Pattern) for name in adata.var_names], dtype=bool)
    id_tmp2 = np.asarray([not str(name).startswith(Gene2Pattern) for name in adata.var_names], dtype=bool)
    id_tmp = np.logical_and(id_tmp1, id_tmp2)
    adata._inplace_subset_var(id_tmp)


def calculate_p(adj, l):
    adj_exp = np.exp(-1 * (adj ** 2) / (2 * (l ** 2)))
    return np.mean(np.sum(adj_exp, 1)) - 1


def search_l(p, adj, start=0.01, end=1000, tol=0.01, max_run=100):
    # binary search l
    run = 0
    p_low = calculate_p(adj, start)
    p_high = calculate_p(adj, end)
    if p_low > p + tol:
        print("l not found, try smaller start point.")
        return None
    elif p_high < p - tol:
        print("l not found, try bigger end point.")
        return None
    elif np.abs(p_low - p) <= tol:
        print("recommended l = ", str(start))
        return start
    elif np.abs(p_high - p) <= tol:
        print("recommended l = ", str(end))
        return end
    while (p_low + tol) < p < (p_high - tol):
        run += 1
        print("Run " + str(run) + ": l [" + str(start) + ", " + str(end) + "], p [" + str(p_low) + ", " + str(
            p_high) + "]")
        if run > max_run:
            print("Exact l not found, closest values are:\n" + "l=" + str(start) + ": " + "p=" + str(
                p_low) + "\nl=" + str(end) + ": " + "p=" + str(p_high))
            return None
        mid = (start + end) / 2
        p_mid = calculate_p(adj, mid)
        if np.abs(p_mid - p) <= tol:
            print("recommended l = ", str(mid))
            return mid
        if p_mid <= p:
            start = mid
            p_low = p_mid
        else:
            end = mid
            p_high = p_mid


def search_res(adata, adj, l, target_num, start=0.4, step=0.1, tol=5e-3, lr=0.05, max_epochs=10, r_seed=100, t_seed=100,
               n_seed=100, max_run=10):
    # move step find res
    random.seed(r_seed)
    torch.manual_seed(t_seed)
    np.random.seed(n_seed)
    res = start
    print("Start at res = ", res, "step = ", step)
    clf = SpaGCN()
    clf.set_l(l)
    clf.train(adata, adj, init_spa=True, init="louvain", res=res, tol=tol, lr=lr, max_epochs=max_epochs)
    y_pred, _ = clf.predict()
    old_num = len(set(y_pred))
    print("Res = ", res, "Num of clusters = ", old_num)
    run = 0
    while old_num != target_num:
        random.seed(r_seed)
        torch.manual_seed(t_seed)
        np.random.seed(n_seed)
        old_sign = 1 if (old_num < target_num) else -1
        clf = SpaGCN()
        clf.set_l(l)
        clf.train(adata, adj, init_spa=True, init="louvain", res=res + step * old_sign, tol=tol, lr=lr,
                  max_epochs=max_epochs)
        y_pred, _ = clf.predict()
        new_num = len(set(y_pred))
        print("Res = ", res + step * old_sign, "Num of clusters = ", new_num)
        if new_num == target_num:
            res = res + step * old_sign
            print("recommended res = ", str(res))
            return res
        new_sign = 1 if (new_num < target_num) else -1
        if new_sign == old_sign:
            res = res + step * old_sign
            print("Res changed to", res)
            old_num = new_num
        else:
            step = step / 2
            print("Step changed to", step)
        if run > max_run:
            print("Exact resolution not found")
            print("Recommended res = ", str(res))
            return res
        run += 1
    print("recommended res = ", str(res))
    return res

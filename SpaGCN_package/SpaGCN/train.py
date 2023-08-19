import csv
import math
import os
import random
import sys

import cv2
import pandas as pd
import scanpy
import torch
import argparse
import pytz
import torch.utils.data as dataloader
import tensorboard_logger
import pickle
import numpy as np
from matplotlib import pyplot as plt

from scanpy import read_10x_h5
from datetime import datetime

from torch.backends import cudnn

from calculate import *


def process_data(opt):
    # concat gene expression and position
    data = read_10x_h5(opt.expression_path)
    position = pd.read_csv(opt.position_path, header=None, na_filter=False, index_col=0)
    data.obs["in_histology"] = position[1]
    data.obs["x_array"] = position[2]
    data.obs["y_array"] = position[3]
    data.obs["x_pixel"] = position[4]
    data.obs["y_pixel"] = position[5]
    # filter and set key
    data = data[data.obs["in_histology"] == 1]
    data.var_names = [name.upper() for name in list(data.var_names)]
    data.var["genename"] = data.var.index.astype("str")
    data.write_h5ad(opt.anndata_path)

    # check
    anndata = scanpy.read(opt.anndata_path)
    img = cv2.imread(opt.histology_path)
    new = img.copy()
    x_pixel = list(anndata.obs["x_pixel"])
    y_pixel = list(anndata.obs["y_pixel"])
    for i in range(len(x_pixel)):
        new[int(x_pixel[i] - 20):int(x_pixel[i] + 20), int(y_pixel[i] - 20):int(y_pixel[i] + 20), :] = 0
    cv2.imwrite(opt.map_path, new)

    # calculate adj by L2 distance
    adj = calculate_adj_matrix(x=x_pixel, y=y_pixel, x_pixel=x_pixel, y_pixel=y_pixel, image=img, beta=opt.beta,
                               alpha=opt.alpha, histology=True)
    np.savetxt(opt.adj_path, adj, delimiter=",")

    # prefilter anndata
    anndata.var_names_make_unique()
    prefilter_genes(anndata, min_cells=3)
    prefilter_specialgenes(anndata)
    scanpy.pp.normalize_per_cell(anndata)
    scanpy.pp.log1p(anndata)
    anndata.write_h5ad(opt.anndata_path)

    return anndata, adj


def calculate_parameters(opt, anndata, adj):
    l = search_l(p=opt.neighbor_p, adj=adj, start=opt.l_start, end=opt.l_end, tol=opt.l_tol, max_run=opt.l_max_run)
    res = search_res(adata=anndata, adj=adj, l=l, target_num=opt.n_clusters, start=opt.res_start, step=opt.res_step,
                     tol=opt.res_tol, lr=opt.res_lr, max_epochs=opt.res_epoch, r_seed=opt.r_seed, t_seed=opt.t_seed,
                     n_seed=opt.n_seed)

    return l, res


def set_data(opt):
    if not os.path.exists(opt.anndata_path) or not os.path.exists(opt.adj_path) or not os.path.exists(
            opt.parameter_path):
        anndata, adj = process_data(opt)
        l, res = calculate_parameters(opt, anndata, adj)
        data = [['l', 'res'], [l, res]]
        with open(opt.parameter_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(data)
        print("l:", l, "res:", res)

    anndata = scanpy.read(opt.anndata_path)
    adj = np.loadtxt(opt.adj_path, delimiter=',')

    with open(opt.parameter_path, 'r', newline='') as file:
        reader = csv.reader(file)
        data = [row for row in reader]
    l = float(data[1][0])
    res = float(data[1][1])

    return anndata, adj, l, res


def train(opt, anndata, adj, l, res):
    random.seed(opt.r_seed)
    torch.manual_seed(opt.t_seed)
    np.random.seed(opt.n_seed)

    model = SpaGCN()
    model.set_l(l)
    model.train(adata=anndata, adj=adj, init_spa=True, init="louvain", res=res, tol=5e-3, lr=opt.learning_rate,
                max_epochs=opt.epochs)
    y_pred, prob = model.predict()
    anndata.obs["pred"] = y_pred
    anndata.obs["pred"] = anndata.obs["pred"].astype('category')
    adj_2d = calculate_adj_matrix(x=list(anndata.obs["x_array"]), y=list(anndata.obs["y_array"]), histology=False)
    refined_pred = refine(sample_id=anndata.obs.index.tolist(), pred=anndata.obs["pred"].tolist(), dis=adj_2d,
                          shape="hexagon")
    anndata.obs["refined_pred"] = refined_pred
    anndata.obs["refined_pred"] = anndata.obs["refined_pred"].astype('category')
    anndata.write_h5ad(os.path.join(opt.save_file_path, 'result.h5ad'))


def parser_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_name', type=str, default='t1')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--save_path', type=str, default='./output')

    parser.add_argument('--expression_path', type=str, default='../data/expression_matrix.h5')
    parser.add_argument('--position_path', type=str, default='../data/positions.txt')
    parser.add_argument('--histology_path', type=str, default='../data/histology.tif')
    parser.add_argument('--anndata_path', type=str, default='../data/anndata.h5ad')
    parser.add_argument('--map_path', type=str, default='../data/map.jpg')
    parser.add_argument('--adj_path', type=str, default='../data/adj.csv')

    parser.add_argument('--neighbor_p', type=float, default=0.5)
    parser.add_argument('--l_start', type=float, default=0.01)
    parser.add_argument('--l_end', type=float, default=1000)
    parser.add_argument('--l_tol', type=float, default=0.01)
    parser.add_argument('--l_max_run', type=int, default=100)

    parser.add_argument('--n_clusters', type=int, default=7)
    parser.add_argument('--res_start', type=float, default=0.7)
    parser.add_argument('--res_step', type=float, default=0.1)
    parser.add_argument('--res_tol', type=float, default=5e-3)
    parser.add_argument('--res_lr', type=float, default=0.05)
    parser.add_argument('--res_epoch', type=int, default=20)
    parser.add_argument('--r_seed', type=int, default=100)
    parser.add_argument('--t_seed', type=int, default=100)
    parser.add_argument('--n_seed', type=int, default=100)

    parser.add_argument('--print_freq', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--val_ratio', type=float, default=0)
    parser.add_argument('--workers', type=int, default=0)

    parser.add_argument('--learning_rate', type=float, default=0.05)
    parser.add_argument('--lr_decay_rate', type=float, default=0.1)
    parser.add_argument('--tolerance', type=float, default=5e-3)
    parser.add_argument('--beta', type=float, default=49)
    parser.add_argument('--alpha', type=float, default=1)

    opt = parser.parse_args()

    train_name = datetime.now(tz=pytz.timezone('Asia/Shanghai')).strftime("%Y%m%d_%H%M%S")
    train_name = train_name + opt.test_name
    opt.save_file_path = os.path.join(opt.save_path, train_name)
    if not os.path.exists(opt.save_file_path):
        os.makedirs(opt.save_file_path)
    opt.parameter_path = '../data/' + opt.test_name + '.csv'

    return opt


if __name__ == '__main__':
    opt = parser_opt()
    anndata, adj, l, res = set_data(opt)
    train(opt, anndata, adj, l, res)



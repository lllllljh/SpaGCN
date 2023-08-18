import csv
import math
import os
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

from scanpy import read_10x_h5
from datetime import datetime
from calculate import *


# def set_model(opt):
#     if torch.cuda.is_available():
#         device = torch.device(opt.device)
#         cudnn.benchmark = True
#     else:
#         device = torch.device("cpu")
#     model = GAT_GCN().to(device)
#     criterion = ListNetLoss()
#
#     return model, criterion
#
#
# def collate_fn(batch_list):
#     batch = []
#     for data_list in batch_list:
#         batch.append(Batch.from_data_list(data_list))
#
#     return batch
#

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

    with open(opt.parameter_path, 'r', newline='') as file:
        reader = csv.reader(file)
        data = [row for row in reader]

    l = data[1][0]
    res = data[1][1]

    return l, res

#
#
# def set_optimizer(opt, model):
#     optimizer = optim.Adam(model.parameters(),
#                            lr=opt.learning_rate)
#
#     return optimizer
#
#
# def save_model(model, optimizer, opt, epoch, save_file):
#     print('Saving...')
#     state = {
#         'opt': opt,
#         'model': model.state_dict(),
#         'optimizer': optimizer.state_dict(),
#         'epoch': epoch,
#     }
#     torch.save(state, save_file)
#     del state
#
#
# def train(train_drug_loader, train_target_loader, model, criterion, optimizer, epoch, opt):
#     model.train()
#     losses_drug = AverageMeter()
#     losses_target = AverageMeter()
#
#     for i, batch_list in enumerate(train_target_loader):
#         loss_total = 0
#         optimizer.zero_grad()
#         for data in batch_list:
#             data.to(opt.device)
#             outs = model(data)
#             loss = criterion(outs, data.y.view(-1, 1).float().to(opt.device))
#             loss = loss / len(batch_list) * 0.5
#             loss_total += loss.item()
#             loss.backward()
#         losses_target.update(loss_total, 1)
#         optimizer.step()
#
#         if i % opt.print_freq == 0:
#             print('Train_Target epoch: {} [ ({:.0f}%)] \tLoss.avg: {:.6f}'.format(epoch,
#                                                                                   100. * i / len(train_target_loader),
#                                                                                   losses_target.avg))
#
#     for i, batch_list in enumerate(train_drug_loader):
#         loss_total = 0
#         optimizer.zero_grad()
#         for data in batch_list:
#             data.to(opt.device)
#             outs = model(data)
#             loss = criterion(outs, data.y.view(-1, 1).float().to(opt.device))
#             loss = loss / len(batch_list) * 0.5
#             loss_total += loss.item()
#             loss.backward()
#         losses_drug.update(loss_total, 1)
#         optimizer.step()
#
#         if i % opt.print_freq == 0:
#             print('Train_Drug epoch: {} [ ({:.0f}%)] \tLoss.avg: {:.6f}'.format(epoch,
#                                                                                 100. * i / len(train_drug_loader),
#                                                                                 losses_drug.avg))
#
#     torch.cuda.empty_cache()
#
#     return losses_drug.avg, losses_target.avg
#
#
# def validate(val_loader, model, opt):
#     model.eval()
#     total_preds = torch.Tensor()
#     total_labels = torch.Tensor()
#     with torch.no_grad():
#         for i, batch_list in enumerate(val_loader):
#             for data in batch_list:
#                 data.to(opt.device)
#                 outs = model(data)
#                 total_preds = torch.cat((total_preds, outs.cpu()), 0)
#                 total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
#
#     return total_labels.numpy().flatten(), total_preds.numpy().flatten()
#
#
# def test(val_loader, model, opt):
#     model.eval()
#     ci_list = []
#     pearson_list = []
#     with torch.no_grad():
#         for i, batch_list in enumerate(val_loader):
#             for data in batch_list:
#                 data.to(opt.device)
#                 outs = model(data)
#                 y = data.y.view(-1, 1).cpu().numpy().flatten()
#                 f = outs.cpu().numpy().flatten()
#                 if len(set(y)) != 1:
#                     ci_val = ci(y, f)
#                     pearson_val = pearson(y, f)
#                     ci_list.append(ci_val)
#                     pearson_list.append(pearson_val)
#     torch.cuda.empty_cache()
#
#     return ci_list, pearson_list
#
#
# def valloss(val_loader, model, criterion, opt):
#     model.eval()
#     losses = AverageMeter()
#     with torch.no_grad():
#         for i, batch_list in enumerate(val_loader):
#             loss_total = 0
#             for data in batch_list:
#                 data.to(opt.device)
#                 outs = model(data)
#                 loss = criterion(outs, data.y.view(-1, 1).float().to(opt.device))
#                 loss = loss / len(batch_list)
#                 loss_total += loss.item()
#             losses.update(loss_total, 1)
#         torch.cuda.empty_cache()
#
#     return losses.avg
#
#
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
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--val_ratio', type=float, default=0)
    parser.add_argument('--workers', type=int, default=0)

    parser.add_argument('--learning_rate', type=float, default=0.0005)
    parser.add_argument('--lr_decay_rate', type=float, default=0.1)
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
    l, res = set_data(opt)

    # model, criterion = set_model(opt)
    # optimizer = set_optimizer(opt, model)
    # logger = tensorboard_logger.Logger(logdir=opt.save_file_path, flush_secs=2)

    #
    # # best_mse = 1000
    # # best_test_mse = 1000
    # # best_ci = 0
    #
    # best_test_ci = 0
    # best_test_pearson = -1
    # best_epoch = -1
    # model_file_name = os.path.join(opt.save_file_path, 'best.pth')
    # result_file_name = os.path.join(opt.save_file_path, 'result.csv')
    #
    # for epoch in range(1, opt.epochs + 1):
    #     loss_drug, loss_target = train(train_drug_loader, train_target_loader, model, criterion, optimizer, epoch, opt)
    #     print('train drug data loss:', loss_drug)
    #     print('train target data loss:', loss_target)
    #     logger.log_value('train_drug_loss', loss_drug, epoch)
    #     logger.log_value('train_target_loss', loss_target, epoch)
    #
    #     # test_loss = valloss(test_loader, model, criterion, opt)
    #     # print('test data loss:', test_loss)
    #     # logger.log_value('test_loss', test_loss, epoch)
    #     #
    #     # G, P = validate(val_loader, model, opt)
    #     # val = ci(G, P)
    #     # logger.log_value('ci', val, epoch)
    #     # print('predicting for valid data CI: ', val)
    #
    #     G, P = validate(test_loader, model, opt)
    #     ci_val = ci(G, P)
    #     pearson_val = pearson(G, P)
    #     logger.log_value('test_ci', ci_val, epoch)
    #     logger.log_value('test_pearson', pearson_val, epoch)
    #     print('test_ci', ci_val, '; test_pearson', pearson_val)
    #     if ci_val > best_test_ci:
    #         best_test_ci = ci_val
    #         best_test_pearson = pearson_val
    #         best_epoch = epoch
    #         print('ci improved at epoch ', best_epoch, '; best_test_ci:', best_test_ci)
    #         print('ci improved at epoch ', best_epoch, '; best_test_pearson:', best_test_pearson)
    #         save_model(model, optimizer, opt, opt.epochs, model_file_name)
    #     else:
    #         print('No improvement since epoch ', best_epoch, '; best_test_ci:', best_test_ci)
    #         print('No improvement since epoch ', best_epoch, '; best_test_pearson:', best_test_pearson)
    #     # if val > best_ci:
    #     #     best_ci = val
    #     #     best_epoch = epoch
    #     #     save_model(model, optimizer, opt, opt.epochs, model_file_name)
    #     #     print('predicting for test data')
    #     #     G, P = validate(test_loader, model, opt)
    #     #     ret = [rmse(G, P), mse(G, P), pearson(G, P), spearman(G, P), ci(G, P)]
    #     #     with open(result_file_name, 'w') as f:
    #     #         f.write(','.join(map(str, ret)))
    #     #     best_test_ci = ci(G, P)
    #     #     print('CI improved at epoch ', best_epoch, '; best_test_ci:', best_test_ci)
    #     # else:
    #     #     print(ret[-1], 'No improvement since epoch ', best_epoch, '; best_test_ci:', best_test_ci)
    #
    # save_file = os.path.join(opt.save_file_path, 'last.pth')
    # save_model(model, optimizer, opt, opt.epochs, save_file)

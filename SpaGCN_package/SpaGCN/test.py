import csv
import pandas as pd
import scanpy
import os
from matplotlib import pyplot as plt
from sklearn.metrics import adjusted_rand_score


def plot(output_path):
    anndata = scanpy.read(os.path.join(output_path, 'result.h5ad'))
    # adata.obs should contain two columns for x_pixel and y_pixel
    # Set colors used
    plot_color = ["#F56867", "#FEB915", "#C798EE", "#59BE86", "#7495D3", "#D1D1D1", "#6D1A9C", "#15821E", "#3A84E6",
                  "#997273", "#787878", "#DB4C6C", "#9E7A7A", "#554236", "#AF5F3C", "#93796C", "#F9BD3F", "#DAB370",
                  "#877F6C", "#268785"]
    # Plot spatial domains
    domains = "pred"
    num_celltype = len(anndata.obs[domains].unique())
    anndata.uns[domains + "_colors"] = list(plot_color[:num_celltype])
    ax = scanpy.pl.scatter(anndata, alpha=1, x="y_pixel", y="x_pixel", color=domains, title=domains,
                           color_map=plot_color,
                           show=False, size=100000 / anndata.shape[0])
    ax.set_aspect('equal', 'box')
    ax.axes.invert_yaxis()
    plt.savefig(os.path.join(output_path, 'pred.png'), dpi=600)
    plt.close()

    # Plot refined spatial domains
    domains = "refined_pred"
    num_celltype = len(anndata.obs[domains].unique())
    anndata.uns[domains + "_colors"] = list(plot_color[:num_celltype])
    ax = scanpy.pl.scatter(anndata, alpha=1, x="y_pixel", y="x_pixel", color=domains, title=domains,
                           color_map=plot_color,
                           show=False, size=100000 / anndata.shape[0])
    ax.set_aspect('equal', 'box')
    ax.axes.invert_yaxis()
    plt.savefig(os.path.join(output_path, 'refined_pred.png'), dpi=600)
    plt.close()


def calculate(output_path, ground_path):
    anndata = scanpy.read_h5ad(os.path.join(output_path, 'result.h5ad'))
    true_labels_map = {"Layer_1": 0, "Layer_2": 1, "Layer_3": 2, "Layer_4": 3, "Layer_5": 4, "Layer_6": 5, "WM": 6}
    ground = pd.read_csv(ground_path, header=None)
    ground = ground.dropna()
    ground[1] = ground[1].map(true_labels_map)
    true = ground[1].to_numpy()
    pred = [anndata.obs['pred'][key] for key in ground[0]]
    refined_pred = [anndata.obs['refined_pred'][key] for key in ground[0]]
    ARI = adjusted_rand_score(true, pred)
    refined_ARI = adjusted_rand_score(true, refined_pred)

    print('ARI:', ARI, ' refined_ARI:', refined_ARI)


if __name__ == '__main__':
    output_path = "./output/20230819_115139t1"
    ground_path = "../data/151673_truth.csv"
    # plot(output_path)
    calculate(output_path, ground_path)


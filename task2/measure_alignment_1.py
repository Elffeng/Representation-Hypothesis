import argparse
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
import umap.umap_ as umap
from matplotlib import pyplot as plt
from sklearn.cross_decomposition import CCA
import torchaudio.functional as TAF
from transform import  *
from align import AlignmentMetrics


def load_rep(r_file):
    hidden_states = torch.load(r_file, weights_only=True)["feats"]
    # feats = metrics.remove_outliers(hidden_states.float(), q=q, exact=exact)
    return hidden_states


# 添加新的分析函数
def analyze_alignment_with_transform(rep_A, rep_B, transform_method=None, transform_params=None,
                                     alignment_method="mutual_knn", alignment_params=None):
    """分析经过变换后的表征对齐度"""
    transform_classes = {
        'scaling': ScalingTransform,
        'rotation': RotationTransform,
        'normalization': NormalizationTransform,
        'orthogonal': OrthogonalTransform,
        'projection': ProjectionTransform,
        'nonlinear': NonlinearTransform,
        'permutation': PermutationTransform
    }

    # 应用变换(如果指定)
    if transform_method is not None:
        transform_params = transform_params or {}
        transform_class = transform_classes[transform_method]

        transformer_A = transform_class(rep_A, **transform_params)
        transformer_B = transform_class(rep_B, **transform_params)

        rep_A = transformer_A.transform()
        rep_B = transformer_B.transform()

    # 计算对齐度
    alignment_params = alignment_params or {'topk': 5}
    metrics = AlignmentMetrics(
        metric=alignment_method,
        rep_left=rep_A,
        rep_right=rep_B,
        **alignment_params
    )

    return metrics.compute_score()


def visualize_different_layers(llm_rep, lvm_rep, args):
    # # 可视化1
    left_model_layers = llm_rep.shape[1]
    right_model_layers = lvm_rep.shape[1]
    alignment_scores = np.zeros((left_model_layers, right_model_layers))
    if args.layer_mode == 'all':
        for i in range(left_model_layers):
            for j in range(right_model_layers):
                args.layer_idx_left = i
                args.layer_idx_right = j
                align_metrics = AlignmentMetrics(metric=args.metric, rep_left=llm_rep[:, args.layer_idx_left],
                                                 rep_right=lvm_rep[:, args.layer_idx_right], topk=10)
                align_score = align_metrics.compute_score()
                alignment_scores[i, j] = align_score
                print(i, j, align_score)
    # 创建图形
    plt.figure(figsize=(8, 6))

    # 绘制热力图，origin='lower'使y轴从下到上
    plt.imshow(alignment_scores, cmap='viridis', aspect='auto', origin='lower')

    # 添加颜色条
    plt.colorbar(label='Alignment Score')

    # 添加标签和标题
    plt.title('Layer Alignment Heatmap')
    plt.xlabel('Right Model Layers')
    plt.ylabel('Left Model Layers')

    # 设置x轴刻度（从0开始）
    plt.xticks(np.arange(right_model_layers))

    # 设置y轴刻度（从0开始）
    plt.yticks(np.arange(left_model_layers))
    print(args.left_model_name)
    print(
        args.right_model_name
    )
    model_name = f"{args.left_model_name}_{args.right_model_name}"
    model_name = model_name.replace("/", "_")

    plt.savefig(f'results/figures/figure1_{args.dataset}_{args.language}_{model_name}_{args.metric}.png')
    # 显示热力图
    plt.show()


def visualize_inner_layers(representation):
    model_layers = representation.shape[1]
    # 可视化3: 两个模型的不同层的alignment score
    alignment_scores = np.zeros(model_layers)
    print(f"Shape of llm_rep: {llm_rep.shape}")
    for i in range(model_layers):
        if i >= representation.shape[1]:
            print(f"Index {i} out of bounds for dimension 1 with size {representation.shape[1]}")
            break
        align_metrics = AlignmentMetrics(metric=args.metric, rep_left=representation[:, i],
                                         rep_right=representation[:, -1], topk=5)
        align_score = align_metrics.compute_score()
        alignment_scores[i] = align_score
        print(i, align_score)
        print(f"Current index (i): {i}")
    # 创建图形
    plt.figure(figsize=(10, 6))

    # 创建x轴的值（将层索引转换为比例）
    x = np.array(range(model_layers)) / (model_layers - 1)
    y = alignment_scores

    # 绘制散点图
    plt.scatter(x, y, color='blue', alpha=0.6, s=50)

    # 使用numpy的polyfit进行线性拟合
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)

    # 绘制拟合直线
    plt.plot(x, p(x), color='blue', alpha=0.8, linewidth=2)

    # 添加网格
    plt.grid(True, linestyle='--', alpha=0.3)

    # 设置标签和标题
    plt.xlabel('Layer Index Ratio')
    plt.ylabel('Alignment Score')
    plt.title('Layer-wise Alignment Scores')

    # 设置x轴的刻度
    plt.xticks(np.arange(0, 1.1, 0.1))

    # 调整布局
    plt.tight_layout()

    model_name = f"{args.left_model_name}_{args.right_model_name}"
    model_name = model_name.replace("/", "_")

    # 显示图形
    plt.savefig(f'results/figures/figure2_{args.dataset}_{args.language}_{model_name}_{args.metric}.png')
    plt.show()


def visualization_metrics_correlation(llm_rep, lvm_rep):
    left_model_layers = llm_rep.shape[1]
    right_model_layers = lvm_rep.shape[1]
    # 可视化4: 评估指标相关性，敏感性等
    metrics = ["cycle_knn", "mutual_knn",
               "cka", "unbiased_cka", "cknna", "svcca",
               "cknna"]
    align_metric_score = defaultdict(list)
    for metric in metrics:
        scores = []
        for i in range(left_model_layers):
            for j in range(right_model_layers):
                align_metrics = AlignmentMetrics(metric=metric, rep_left=llm_rep[:, i],
                                                 rep_right=lvm_rep[:, j], topk=5)
                align_score = align_metrics.compute_score()
                scores.append(align_score)
        scores = np.array(scores)
        normalized_score = (scores - scores.min()) / (scores.max() - scores.min())
        align_metric_score[metric] = normalized_score

    scores_df = pd.DataFrame(align_metric_score)
    correlation_matrices = {
        # 'pearson': scores_df.corr(method='pearson'),
        'spearman': scores_df.corr(method='spearman'),
        # 'kendall': scores_df.corr(method='kendall')
    }
    fig, axes = plt.subplots(1, 1, figsize=(8, 6))

    # for i, (corr_type, corr_matrix) in enumerate(correlation_matrices.items()):
    sns.heatmap(correlation_matrices['spearman'],
                annot=True,
                cmap='RdYlBu_r',
                vmin=-1,
                vmax=1,
                ax=axes,
                fmt='.2f')

    axes.set_title(f'{"spearman".capitalize()} Correlation')

    plt.tight_layout()

    model_name = f"{args.left_model_name}_{args.right_model_name}"
    model_name = model_name.replace("/", "_")

    plt.savefig(f'results/figures/figure3_{args.dataset}_{args.language}_{model_name}_{args.metric}.png')
    plt.show()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="multi30k", choices=['multi30k'])
    parser.add_argument("--metric", type=str, default="mutual_knn", nargs='+',
                        choices=["cycle_knn", "mutual_knn", "lcs_knn",
                                 "cka", "unbiased_cka", "cknna", "svcca",
                                 "edit_distance_knn", "cknna"])
    # cka
    parser.add_argument("--subset", type=str, default="wit_1024")
    parser.add_argument("--caption_idx", type=int, default=0)
    parser.add_argument("--layer_idx_left", type=int, default=-1)
    parser.add_argument("--layer_idx_right", type=int, default=-1)
    parser.add_argument("--left_model_name", type=str, default="EleutherAI/pythia-410m", choices=["test"])
    parser.add_argument("--right_model_name", type=str, default="vit_giant_patch14_dinov2.lvd142m", choices=["test"])
    parser.add_argument("--feature_dir", type=str,
                        default="./results/features/")
    parser.add_argument("--layer_mode", type=str, default="all", choices=["all", "last"])
    parser.add_argument("--language", type=str, default="en", choices=["en"])

    # 添加新的参数
    parser.add_argument("--transform_method", type=str, default=None,
                        choices=['scaling', 'rotation', 'normalization',
                                 'orthogonal', 'projection', 'nonlinear', 'permutation'])

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    return args


if __name__ == "__main__":
    args = get_args()

    # 加载表征
    llm_rep = load_rep(
        r_file="/Users/fengdan/Desktop/essay/platonic-rep-main/results/features/multi30k/EleutherAI_pythia-410m_pool-avg.pt")
    lvm_rep = load_rep(
        r_file="/Users/fengdan/Desktop/essay/platonic-rep-main/results/features/multi30k/vit_giant_patch14_dinov2.lvd142m_pool-cls.pt")

    # 如果指定了变换方法，进行变换后的对齐分析
    if args.transform_method:
        score = analyze_alignment_with_transform(
            rep_A=llm_rep,
            rep_B=lvm_rep,
            transform_method=args.transform_method,
            alignment_method=args.metric
        )
        print(f"Alignment score after {args.transform_method} transform: {score}")

    # 保留原有的可视化逻辑
    visualize_different_layers(llm_rep, lvm_rep, args)
    visualize_inner_layers(llm_rep)
    visualize_inner_layers(lvm_rep)
    visualization_metrics_correlation(llm_rep, lvm_rep)
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import seaborn as sns
import torch
import umap.umap_ as umap
from matplotlib import pyplot as plt
from sklearn.cross_decomposition import CCA
import torchaudio.functional as TAF


def load_rep(r_file):
    hidden_states = torch.load(r_file, weights_only=True)["feats"]
    # feats = metrics.remove_outliers(hidden_states.float(), q=q, exact=exact)
    return hidden_states


class AlignmentMetrics:
    SUPPORTED_METRICS = [

    ]

    def __init__(self, metric, rep_left, rep_right, topk):
        self.metric = metric
        self.rep_left = rep_left
        self.rep_right = rep_right
        self.topk = topk

    def compute_score(self):
        if self.metric == "cycle_knn":
            metric_value = self.cycle_knn(self.rep_left, self.rep_right, self.topk)
        elif self.metric == "mutual_knn":
            metric_value = self.mutual_knn(self.rep_left, self.rep_right, self.topk)
        elif self.metric == "lcs_knn":
            metric_value = self.lcs_knn(self.rep_left, self.rep_right, self.topk)
        elif self.metric == "cka":
            metric_value = self.cka(self.rep_left, self.rep_right)
        elif self.metric == "unbiased_cka":
            metric_value = self.unbiased_cka(self.rep_left, self.rep_right)
        elif self.metric == "svcca":
            metric_value = self.svcca(self.rep_left, self.rep_right)
        elif self.metric == "edit_distance_knn":
            metric_value = self.edit_distance_knn(self.rep_left, self.rep_right, self.topk)
        elif self.metric == "cknna":
            metric_value = self.cknna(self.rep_left, self.rep_right, self.topk)
        else:
            raise ValueError(f"Invalid metric {self.metric}")
        return metric_value

    def compute_nearest_neighbors(self, feats, topk=1):
        """
        Compute the nearest neighbors of feats
        Args:
            feats: a torch tensor of shape N x D
            topk: the number of nearest neighbors to return
        Returns:
            knn: a torch tensor of shape N x topk
        """
        assert feats.ndim == 2, f"Expected feats to be 2D, got {feats.ndim}"
        knn = (
            (feats @ feats.T).fill_diagonal_(-1e8).argsort(dim=1, descending=True)[:, :topk]
        )
        return knn

    def compute_knn_accuracy(self, knn):
        """
        Compute the accuracy of the nearest neighbors. Assumes index is the gt label.
        Args:
            knn: a torch tensor of shape N x topk
        Returns:
            acc: a float representing the accuracy
        """
        n = knn.shape[0]
        acc = knn == torch.arange(n, device=knn.device).view(-1, 1, 1)
        acc = acc.float().view(n, -1).max(dim=1).values.mean()
        return acc

    def longest_ordinal_sequence(self, X, Y):
        """ For each pair in X and Y, compute the length of the longest sub-sequence (LCS) """

        def compute_distance(X, Y, dist_fn):
            """ compute distance in parallel"""
            B, N = X.shape
            distances = np.zeros(B)
            X, Y = X.cpu().numpy(), Y.cpu().numpy()

            if pymp_available:
                with pymp.Parallel(4) as p:
                    for i in p.range(B):
                        distances[i] = dist_fn(X[i], Y[i])
            else:
                for i in range(B):
                    distances[i] = dist_fn(X[i], Y[i])
            return torch.tensor(distances)

        def lcs_length(x, y):
            """
            Compute the length of the longest common subsequence between two sequences.
            This is a classic dynamic programming implementation.
            """
            m, n = len(x), len(y)
            dp = [[0] * (n + 1) for _ in range(m + 1)]

            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if x[i - 1] == y[j - 1]:
                        dp[i][j] = dp[i - 1][j - 1] + 1
                    else:
                        dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
            return dp[m][n]

        lcs = compute_distance(X, Y, lcs_length)
        return lcs

    def compute_distance(self, X, Y, dist_fn):
        """ compute distance in parallel"""
        B, N = X.shape
        distances = np.zeros(B)
        X, Y = X.cpu().numpy(), Y.cpu().numpy()

        if pymp_available:
            with pymp.Parallel(4) as p:
                for i in p.range(B):
                    distances[i] = dist_fn(X[i], Y[i])
        else:
            for i in range(B):
                distances[i] = dist_fn(X[i], Y[i])
        return torch.tensor(distances)

    def hsic_unbiased(self, K, L):
        """
        Compute the unbiased Hilbert-Schmidt Independence Criterion (HSIC) as per Equation 5 in the paper.
        > Reference: https://jmlr.csail.mit.edu/papers/volume13/song12a/song12a.pdf
        """
        m = K.shape[0]

        # Zero out the diagonal elements of K and L
        K_tilde = K.clone().fill_diagonal_(0)
        L_tilde = L.clone().fill_diagonal_(0)

        # Compute HSIC using the formula in Equation 5
        HSIC_value = (
                (torch.sum(K_tilde * L_tilde.T))
                + (torch.sum(K_tilde) * torch.sum(L_tilde) / ((m - 1) * (m - 2)))
                - (2 * torch.sum(torch.mm(K_tilde, L_tilde)) / (m - 2))
        )

        HSIC_value /= m * (m - 3)
        return HSIC_value

    def hsic_biased(self, K, L):
        """ Compute the biased HSIC (the original CKA) """
        H = torch.eye(K.shape[0], dtype=K.dtype, device=K.device) - 1 / K.shape[0]
        return torch.trace(K @ H @ L @ H)

    def cycle_knn(self, feats_A, feats_B, topk):
        """
        LLM nearest neighbors -> Query Language Pair -> LVM nearest neighbors
        Args:
            feats_A: A torch tensor of shape N x feat_dim
            feats_B: A torch tensor of shape N x feat_dim

        Returns:
            acc: a float representing the accuracy
        """
        knn_A = self.compute_nearest_neighbors(feats_A, topk)
        knn_B = self.compute_nearest_neighbors(feats_B, topk)
        t = knn_A[knn_B]
        r = self.compute_knn_accuracy(t).item()
        return r

    def mutual_knn(self, feats_A, feats_B, topk):
        """
        Computes the mutual KNN accuracy.

        Args:
            feats_A: A torch tensor of shape N x feat_dim
            feats_B: A torch tensor of shape N x feat_dim

        Returns:
            A float representing the mutual KNN accuracy
        """
        knn_A = self.compute_nearest_neighbors(feats_A, topk)
        knn_B = self.compute_nearest_neighbors(feats_B, topk)

        n = knn_A.shape[0]
        topk = knn_A.shape[1]

        # Create a range tensor for indexing
        range_tensor = torch.arange(n, device=knn_A.device).unsqueeze(1)

        # Create binary masks for knn_A and knn_B
        lvm_mask = torch.zeros(n, n, device=knn_A.device)
        llm_mask = torch.zeros(n, n, device=knn_A.device)

        lvm_mask[range_tensor, knn_A] = 1.0
        llm_mask[range_tensor, knn_B] = 1.0

        acc = lvm_mask * llm_mask
        acc = acc.sum(dim=1)
        acc /= topk

        return acc.mean().item()

    def lcs_knn(self, feats_A, feats_B, topk):
        knn_A = self.compute_nearest_neighbors(feats_A, topk)
        knn_B = self.compute_nearest_neighbors(feats_B, topk)
        score = self.longest_ordinal_sequence(knn_A, knn_B).float().mean()
        return score

    def cka(self, feats_A, feats_B, kernel_metric='ip', rbf_sigma=1.0, unbiased=False):
        """Computes the unbiased Centered Kernel Alignment (CKA) between features."""

        if kernel_metric == 'ip':
            # Compute kernel matrices for the linear case
            K = torch.mm(feats_A, feats_A.T)
            L = torch.mm(feats_B, feats_B.T)
        elif kernel_metric == 'rbf':
            # COMPUTES RBF KERNEL
            K = torch.exp(-torch.cdist(feats_A, feats_A) ** 2 / (2 * rbf_sigma ** 2))
            L = torch.exp(-torch.cdist(feats_B, feats_B) ** 2 / (2 * rbf_sigma ** 2))
        else:
            raise ValueError(f"Invalid kernel metric {kernel_metric}")

        # Compute HSIC values
        hsic_fn = self.hsic_unbiased if unbiased else self.hsic_biased
        hsic_kk = hsic_fn(K, K)
        hsic_ll = hsic_fn(L, L)
        hsic_kl = hsic_fn(K, L)

        # Compute CKA
        # print('hsic', hsic_kl)
        cka_value = hsic_kl / (torch.sqrt(hsic_kk * hsic_ll) + 1e-6)
        return cka_value.item()

    def unbiased_cka(self, feats_A, feats_B, kernel_metric='ip', rbf_sigma=1.0, unbiased=True):
        return self.cka(feats_A, feats_B, unbiased=True)

    def svcca(self, feats_A, feats_B, cca_dim=10):

        # Center and scale the activations
        def preprocess_activations(act):
            act = act - torch.mean(act, axis=0)
            act = act / (torch.std(act, axis=0) + 1e-8)
            return act

        feats_A = preprocess_activations(feats_A)
        feats_B = preprocess_activations(feats_B)

        # Compute SVD
        U1, _, _ = torch.svd_lowrank(feats_A, q=cca_dim)
        U2, _, _ = torch.svd_lowrank(feats_B, q=cca_dim)

        U1 = U1.cpu().detach().numpy()
        U2 = U2.cpu().detach().numpy()

        # Compute CCA
        cca = CCA(n_components=cca_dim)
        cca.fit(U1, U2)
        U1_c, U2_c = cca.transform(U1, U2)

        # sometimes it goes to nan, this is just to avoid that
        U1_c += 1e-10 * np.random.randn(*U1_c.shape)
        U2_c += 1e-10 * np.random.randn(*U2_c.shape)

        # Compute SVCCA similarity
        svcca_similarity = np.mean(
            [np.corrcoef(U1_c[:, i], U2_c[:, i])[0, 1] for i in range(cca_dim)]
        )
        return svcca_similarity

    def edit_distance_knn(self, feats_A, feats_B, topk):
        """
        Computes the edit distance between the nearest neighbors of feats_A and feats_B.
        """
        knn_A = self.compute_nearest_neighbors(feats_A, topk)
        knn_B = self.compute_nearest_neighbors(feats_B, topk)

        # given N x topk with integer entries, compute edit distance
        n = knn_A.shape[0]
        topk = knn_A.shape[1]

        edit_distance = self.compute_distance(knn_A, knn_B, TAF.edit_distance)
        return 1 - torch.mean(edit_distance) / topk

    def cknna(self, feats_A, feats_B, topk=None, distance_agnostic=False, unbiased=True):
        """ similarity only cka variant """
        n = feats_A.shape[0]

        if topk < 2:
            raise ValueError("CKNNA requires topk >= 2")

        if topk is None:
            topk = feats_A.shape[0] - 1

        K = feats_A @ feats_A.T
        L = feats_B @ feats_B.T
        device = feats_A.device

        def similarity(K, L, topk):
            if unbiased:
                K_hat = K.clone().fill_diagonal_(float("-inf"))
                L_hat = L.clone().fill_diagonal_(float("-inf"))
            else:
                K_hat, L_hat = K, L

            # get topk indices for each row
            # if unbiased we cannot attend to the diagonal unless full topk
            # else we can attend to the diagonal
            _, topk_K_indices = torch.topk(K_hat, topk, dim=1)
            _, topk_L_indices = torch.topk(L_hat, topk, dim=1)

            # create masks for nearest neighbors
            mask_K = torch.zeros(n, n, device=device).scatter_(1, topk_K_indices, 1)
            mask_L = torch.zeros(n, n, device=device).scatter_(1, topk_L_indices, 1)

            # intersection of nearest neighbors
            mask = mask_K * mask_L

            if distance_agnostic:
                sim = mask * 1.0
            else:
                if unbiased:
                    sim = self.hsic_unbiased(mask * K, mask * L)
                else:
                    sim = self.hsic_biased(mask * K, mask * L)
            return sim

        sim_kl = similarity(K, L, topk)
        sim_kk = similarity(K, K, topk)
        sim_ll = similarity(L, L, topk)

        return sim_kl.item() / (torch.sqrt(sim_kk * sim_ll) + 1e-6).item()


def compute_alignment(x_feat_paths, y_feat_paths, metric, topk, precise=True):
    pass


def visualization():
    align_scores = [0.1, 0.3, 0.5, 0.6, 0.8]
    align_errors = [0.02, 0.02, 0.02, 0.03, 0.03]
    vtab_tasks_solved_labels = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']
    # plot_alignment_figure(align_scores, align_errors, vtab_tasks_solved_labels)

    plt.figure(figsize=(10, 5))
    colors = ['yellow', 'green', 'lightgreen', 'darkgreen', 'blue']
    plt.bar(vtab_tasks_solved_labels, align_scores, yerr=align_errors, color=colors, alpha=0.7)
    plt.xlabel('Percentage of VTAB Tasks Solved (total=19)')
    plt.ylabel('Intra-bucket Alignment')
    plt.title('Convergence to General Competence')
    plt.show()

    high_dim_feats = np.random.rand(5, 128)
    model_types = ['Classification', 'MAE', 'Random Initialization', 'Contrastive', 'CLIP']
    markers = ['o', 's', '^', '*', 'D']
    vtab_tasks_solved = [3, 6, 9, 15, 19]
    # plot_umap_figure(high_dim_feats, model_types, vtab_tasks_solved, markers)
    umap_reducer = umap.UMAP()
    umap_embeds = umap_reducer.fit_transform(high_dim_feats)

    plt.figure(figsize=(8, 6))
    for i, model_type in enumerate(model_types):
        plt.scatter(umap_embeds[i, 0], umap_embeds[i, 1], c=[vtab_tasks_solved[i]],
                    cmap='coolwarm', s=100, alpha=0.8, label=model_type, marker=markers[i])

    plt.colorbar(label='VTAB Tasks Solved')
    plt.title('UMAP of Model Representations')
    plt.legend(title="Model Types")
    plt.show()


def visualize_different_layers(alignment_scores):
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
    plt.savefig('../results/figures/alignment_scores.png')

    # 显示热力图
    plt.show()

def visualize_alignment_scores(left_model_layers, right_model_layers, llm_rep, lvm_rep, args):
    alignment_scores = np.zeros((left_model_layers, right_model_layers))
    for i in range(left_model_layers):
        for j in range(right_model_layers):
            args.layer_idx_left = i
            args.layer_idx_right = j
            align_metrics = AlignmentMetrics(
                metric=args.metric, 
                rep_left=llm_rep[:, args.layer_idx_left], 
                rep_right=lvm_rep[:, args.layer_idx_right], 
                topk=5
            )
            align_score = align_metrics.compute_score()
            alignment_scores[i, j] = align_score
            print(i, j, align_score)
    visualize_different_layers(alignment_scores)


def visualize_metrics_alignment(llm_rep, lvm_rep, metrics, args):

    scores = []

    for metric in metrics:
        align_metrics = AlignmentMetrics(
            metric=metric, 
            rep_left=llm_rep[:, -1], 
            rep_right=lvm_rep[:, -1], 
            topk=5
        )
        align_score = align_metrics.compute_score()
        scores.append(align_score)
        print(f'Last layer - Metric: {metric}, Alignment Score: {align_score}')

    # 创建折线图
    plt.figure(figsize=(12, 6))

    # 绘制折线图
    plt.plot(metrics, scores, marker='o', linewidth=2, markersize=8)

    # 添加标签和标题
    plt.title('Alignment Scores across Different Metrics')
    plt.xlabel('Metrics')
    plt.ylabel('Alignment Score')

    # 旋转x轴标签以防止重叠
    plt.xticks(rotation=45, ha='right')

    # 添加网格线以提高可读性
    plt.grid(True, linestyle='--', alpha=0.7)

    # 调整布局以确保所有元素可见
    plt.tight_layout()

    # 保存图像
    plt.savefig('../results/figures/alignment_metric_scores.png')

    # 显示图形
    plt.show()


def visualize_inner_layers_alignment(llm_rep, lvm_rep, left_model_layers, right_model_layers, args):
    def visualize_inner_layers(alignment_scores, layer_num, model_name):
        # 创建图形
        plt.figure(figsize=(10, 6))

        # 创建x轴的值（将层索引转换为比例）
        x = np.array(range(layer_num)) / (layer_num - 1)
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
        plt.title(f'{model_name} - Layer-wise Alignment Scores')

        # 设置x轴的刻度
        plt.xticks(np.arange(0, 1.1, 0.1))

        # 调整布局
        plt.tight_layout()

        # 显示图形
        plt.savefig(f'../results/figures/alignment_{model_name}_inner_layers.png')
        plt.show()

    # 可视化左模型（llm_rep）
    alignment_scores = np.zeros(left_model_layers)
    for i in range(left_model_layers):
        align_metrics = AlignmentMetrics(
            metric=args.metric, 
            rep_left=llm_rep[:, i], 
            rep_right=llm_rep[:, -1], 
            topk=5
        )
        align_score = align_metrics.compute_score()
        alignment_scores[i] = align_score
        print(f"Left model - Layer {i}: {align_score}")
    visualize_inner_layers(alignment_scores, left_model_layers, 'left_model')

    # 可视化右模型（lvm_rep）
    alignment_scores = np.zeros(right_model_layers)
    for j in range(right_model_layers):
        align_metrics = AlignmentMetrics(
            metric=args.metric, 
            rep_left=lvm_rep[:, j], 
            rep_right=lvm_rep[:, -1], 
            topk=5
        )
        align_score = align_metrics.compute_score()
        alignment_scores[j] = align_score
        print(f"Right model - Layer {j}: {align_score}")
    visualize_inner_layers(alignment_scores, right_model_layers, 'right_model')

def visualize_metrics_correlation(llm_rep, lvm_rep, left_model_layers, right_model_layers, metrics):
    align_metric_score = defaultdict(list)
    for metric in metrics:
        scores = []
        for i in range(left_model_layers):
            for j in range(right_model_layers):
                align_metrics = AlignmentMetrics(
                    metric=metric, 
                    rep_left=llm_rep[:, i], 
                    rep_right=lvm_rep[:, j], 
                    topk=5
                )
                align_score = align_metrics.compute_score()
                scores.append(align_score)
        
        # 归一化分数
        scores = np.array(scores)
        normalized_score = (scores - scores.min()) / (scores.max() - scores.min())
        align_metric_score[metric] = normalized_score
    
    # 将计算出来的对齐指标分数转换为DataFrame
    scores_df = pd.DataFrame(align_metric_score)
    
    # 计算Spearman相关性矩阵
    correlation_matrices = {
        'spearman': scores_df.corr(method='spearman'),
    }
    
    # 创建一个热图来展示相关性
    fig, axes = plt.subplots(1, 1, figsize=(8, 6))
    
    sns.heatmap(correlation_matrices['spearman'],
                annot=True,
                cmap='RdYlBu_r',
                vmin=-1,
                vmax=1,
                ax=axes,
                fmt='.2f')

    axes.set_title(f'Spearman Correlation')

    plt.tight_layout()
    plt.savefig("../results/figures/metric_correlation.png")
    plt.show()

def visualize_metric_clusters(align_metric_score):
    scores_df = pd.DataFrame(align_metric_score)
    correlation_matrices = {
        'spearman': scores_df.corr(method='spearman'),
    }
    clustered_metrics = {
        'high_correlation': [],  
        'moderate_correlation': [],  
        'low_correlation': []  
    }

    spearman_corr = correlation_matrices['spearman']
    n_metrics = len(spearman_corr)
    processed_pairs = set()

    for i in range(n_metrics):
        for j in range(i + 1, n_metrics):
            if (i, j) not in processed_pairs:
                corr = abs(spearman_corr.iloc[i, j])
                pair = (spearman_corr.index[i], spearman_corr.columns[j])

                if corr > 0.8:
                    clustered_metrics['high_correlation'].append((pair, corr))
                elif corr > 0.5:
                    clustered_metrics['moderate_correlation'].append((pair, corr))
                else:
                    clustered_metrics['low_correlation'].append((pair, corr))

                processed_pairs.add((i, j))
                processed_pairs.add((j, i))
    print("==== 指标相关性分析结果 ====\n")

    print("高度相关的指标对 (|r| > 0.8):")
    for pair, corr in sorted(clustered_metrics['high_correlation'], key=lambda x: -x[1]):
        print(f"  {pair[0]} - {pair[1]}: {corr:.3f}")

    print("\n中度相关的指标对 (0.5 < |r| <= 0.8):")
    for pair, corr in sorted(clustered_metrics['moderate_correlation'], key=lambda x: -x[1]):
        print(f"  {pair[0]} - {pair[1]}: {corr:.3f}")

    print("\n低度相关的指标对 (|r| <= 0.5):")
    for pair, corr in sorted(clustered_metrics['low_correlation'], key=lambda x: -x[1]):
        print(f"  {pair[0]} - {pair[1]}: {corr:.3f}")

def get_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--force_download", action="store_true")
    # parser.add_argument("--force_remake", action="store_true")
    # parser.add_argument("--num_samples", type=int, default=1024)
    # parser.add_argument("--batch_size", type=int, default=4)
    # parser.add_argument("--pool", type=str, default='avg', choices=['avg', 'cls'])
    # parser.add_argument("--prompt", action="store_true")
    parser.add_argument("--dataset", type=str, default="minhuh/prh")
    parser.add_argument("--metric", type=str, default="mutual_knn", nargs='+',
                        choices=["cycle_knn", "mutual_knn", "lcs_knn",
                                 "cka", "unbiased_cka", "cknna", "svcca",
                                 "edit_distance_knn", "cknna"])
    parser.add_argument("--subset", type=str, default="wit_1024")
    parser.add_argument("--caption_idx", type=int, default=0)
    parser.add_argument("--layer_idx_left", type=int, default=-1)
    parser.add_argument("--layer_idx_right", type=int, default=-1)
    # parser.add_argument("--modelset", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--left_model_name", type=str, default="bigscience/bloomz-560m", choices=["val", "test"])
    parser.add_argument("--right_model_name", type=str, default="bigscience/bloomz-560m", choices=["val", "test"])
    # parser.add_argument("--modality", type=str, default="language", choices=["vision", "language", "all"])
    parser.add_argument("--output_dir", type=str, default="./results/features")
    parser.add_argument("--layer_mode", type=str, default="last", choices=["all", "last"])
    # parser.add_argument("--qlora", action="store_true")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    return args


if __name__ == "__main__":
    args = get_args()

    llm_rep = load_rep(r_file="../results/features/flowers102/bigscience_bloomz-560m_pool-avg.pt")
    lvm_rep = load_rep(r_file="../results/features/flowers102/vit_tiny_patch16_224.augreg_in21k_pool-cls.pt")

    left_model_layers = llm_rep.shape[1]
    right_model_layers = lvm_rep.shape[1]

    print(f'Tested model: {args.left_model_name}, {args.right_model_name}, metric: {args.metric}')

    # 可视化1: 不同层的alignment score
    visualize_alignment_scores(left_model_layers, right_model_layers, llm_rep, lvm_rep, args)

    # 可视化2: 不同指标的alignment score
    metrics = ["cycle_knn", "mutual_knn", "cka", "unbiased_cka", "cknna", "svcca", "cknna"]
    visualize_metrics_alignment(llm_rep, lvm_rep, metrics, args)

    # 可视化3: 不同层与最后一层的alignment score
    visualize_inner_layers_alignment(llm_rep, lvm_rep, left_model_layers, right_model_layers, args)

    # 可视化4: 评估指标相关性
    visualize_metrics_correlation(llm_rep, lvm_rep, left_model_layers, right_model_layers, metrics)

    # 可视化5: 指标聚类分析
    align_metric_score = defaultdict(list)
    visualize_metric_clusters(align_metric_score)

    



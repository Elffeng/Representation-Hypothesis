import torch
import torch.nn.functional as F


# 修改原有的AlignmentMetrics类，添加新的对齐方法
class AlignmentMetrics:
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
        # 新增的对齐方法
        elif self.metric == "matrix_multiply":
            metric_value = self.matrix_multiply(self.rep_left, self.rep_right)
        elif self.metric == "cosine":
            metric_value = self.cosine_similarity(self.rep_left, self.rep_right)
        elif self.metric == "procrustes":
            metric_value = self.procrustes_analysis(self.rep_left, self.rep_right)
        elif self.metric == "mutual_info":
            metric_value = self.mutual_information(self.rep_left, self.rep_right)
        else:
            raise ValueError(f"Invalid metric {self.metric}")
        return metric_value

    # [保留原有的所有方法...]

    # 添加新的对齐方法
    def matrix_multiply(self, feats_A, feats_B):
        """矩阵乘法对齐度量"""
        feats_A = F.normalize(feats_A, p=2, dim=1)
        feats_B = F.normalize(feats_B, p=2, dim=1)
        similarity = torch.mm(feats_A, feats_B.t())
        return torch.mean(torch.diagonal(similarity)).item()

    def cosine_similarity(self, feats_A, feats_B):
        """余弦相似度对齐度量"""
        cos_sim = F.cosine_similarity(feats_A, feats_B)
        return torch.mean(cos_sim).item()

    def procrustes_analysis(self, feats_A, feats_B):
        """Procrustes分析对齐度量"""
        feats_A = feats_A - torch.mean(feats_A, dim=0)
        feats_B = feats_B - torch.mean(feats_B, dim=0)
        U, _, V = torch.svd(torch.mm(feats_B.t(), feats_A))
        R = torch.mm(U, V.t())
        aligned_B = torch.mm(feats_B, R)
        return -torch.norm(feats_A - aligned_B).item()

    def mutual_information(self, feats_A, feats_B, n_bins=50):
        """互信息对齐度量"""

        def estimate_mi(x, y, n_bins):
            x_bins = torch.histc(x, bins=n_bins)
            y_bins = torch.histc(y, bins=n_bins)
            px = x_bins / torch.sum(x_bins)
            py = y_bins / torch.sum(y_bins)
            xy_hist = torch.histogramdd(
                torch.stack([x, y], dim=1),
                bins=n_bins
            )[0]
            pxy = xy_hist / torch.sum(xy_hist)
            mi = 0
            for i in range(n_bins):
                for j in range(n_bins):
                    if pxy[i, j] > 0:
                        mi += pxy[i, j] * torch.log(pxy[i, j] / (px[i] * py[j]))
            return mi

        total_mi = 0
        for i in range(feats_A.shape[1]):
            total_mi += estimate_mi(feats_A[:, i], feats_B[:, i], n_bins)
        return (total_mi / feats_A.shape[1]).item()

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



# 首先定义所有表征变换类
class ScalingTransform:
    """缩放变换：可以进行uniform或feature-wise缩放

    这个变换可以帮助我们理解表征的尺度敏感性。uniform scaling保持特征之间的相对关系，
    而feature-wise scaling则可能改变特征间的相对重要性。
    """

    def __init__(self, rep, scale_factor=2.0, mode='uniform'):
        self.rep = rep
        self.scale_factor = scale_factor
        self.mode = mode

    def transform(self):
        if self.mode == 'uniform':
            return self.rep * self.scale_factor
        else:
            scale_factors = torch.rand(self.rep.shape[1]) * self.scale_factor
            return self.rep * scale_factors


class RotationTransform:
    """旋转变换：通过随机正交矩阵进行旋转

    旋转变换可以帮助我们理解表征在不同坐标系下的行为。如果表征的性质在旋转后保持不变，
    说明这些性质是旋转不变的。
    """

    def __init__(self, rep, angle=None):
        self.rep = rep
        self.angle = angle

    def transform(self):
        dim = self.rep.shape[1]
        if self.angle is None:
            random_matrix = torch.randn(dim, dim)
            q, _ = torch.linalg.qr(random_matrix)
            return torch.mm(self.rep, q)
        else:
            rotation_matrix = torch.tensor([
                [torch.cos(self.angle), -torch.sin(self.angle)],
                [torch.sin(self.angle), torch.cos(self.angle)]
            ])
            return torch.mm(self.rep, rotation_matrix)


class NormalizationTransform:
    """标准化变换：支持z-score、min-max和L2标准化

    标准化可以消除不同特征之间的尺度差异，使得所有特征在相似的范围内变化。
    不同的标准化方法适用于不同的场景。
    """

    def __init__(self, rep, mode='z_score'):
        self.rep = rep
        self.mode = mode

    def transform(self):
        if self.mode == 'z_score':
            mean = torch.mean(self.rep, dim=0)
            std = torch.std(self.rep, dim=0)
            return (self.rep - mean) / (std + 1e-8)
        elif self.mode == 'min_max':
            min_val = torch.min(self.rep, dim=0).values
            max_val = torch.max(self.rep, dim=0).values
            return (self.rep - min_val) / (max_val - min_val + 1e-8)
        elif self.mode == 'l2':
            return F.normalize(self.rep, p=2, dim=1)


class OrthogonalTransform:
    """正交变换：通过SVD保持正交性

    正交变换保持向量之间的角度和距离关系，这对于研究表征的几何结构很有帮助。
    """

    def __init__(self, rep):
        self.rep = rep

    def transform(self):
        U, _, V = torch.svd(self.rep)
        return torch.mm(U, V.t())


class ProjectionTransform:
    """投影变换：支持PCA和随机投影

    投影变换可以帮助我们研究表征在低维空间中的行为，以及它们的主要变化方向。
    """

    def __init__(self, rep, n_components=None, method='pca'):
        self.rep = rep
        self.n_components = n_components or rep.shape[1] // 2
        self.method = method

    def transform(self):
        if self.method == 'pca':
            U, _, V = torch.svd(self.rep)
            return torch.mm(self.rep, V[:, :self.n_components])
        elif self.method == 'random':
            proj_matrix = torch.randn(self.rep.shape[1], self.n_components)
            proj_matrix = F.normalize(proj_matrix, p=2, dim=0)
            return torch.mm(self.rep, proj_matrix)


class NonlinearTransform:
    """非线性变换：包括RBF核和多项式变换

    非线性变换可以帮助我们研究表征在非线性映射下的行为，这对于理解表征的非线性特性很重要。
    """

    def __init__(self, rep, method='rbf', gamma=1.0):
        self.rep = rep
        self.method = method
        self.gamma = gamma

    def transform(self):
        if self.method == 'rbf':
            sq_dists = torch.cdist(self.rep, self.rep, p=2).pow(2)
            return torch.exp(-self.gamma * sq_dists)
        elif self.method == 'polynomial':
            return torch.pow(self.rep, 2)


class PermutationTransform:
    """置换变换：随机重排特征维度

    置换变换可以帮助我们理解特征顺序对表征的影响，以及表征的排列不变性。
    """

    def __init__(self, rep):
        self.rep = rep

    def transform(self):
        perm = torch.randperm(self.rep.shape[1])
        return self.rep[:, perm]


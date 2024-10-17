import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
import metrics
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def anchor_and_similarity(x_feats, y_feats, anchor_ratio=0.1, similarity_metric="cosine", topk=10, anchor_strategy="random"):
    """
    Combines anchor selection and similarity computation into one function.
    
    Args:
        x_feats: A torch tensor of shape N x L x D (language model features)
        y_feats: A torch tensor of shape N x L x D (vision model features)
        anchor_ratio: The ratio of the total data to be selected as anchors
        similarity_metric: The similarity metric to be used ("cosine", "euclidean", etc.)
        topk: Number of nearest neighbors for KNN
        anchor_strategy: Strategy for anchor selection ("random", "fps", "kmeans", "topk")
    
    Returns:
        score: The alignment score based on the anchor selection and similarity computation
    """
    
    # Anchor selection based on the strategy
    num_samples = x_feats.shape[0]
    num_anchors = int(num_samples * anchor_ratio)
    
    if anchor_strategy == "random":
        # Uniform random selection of anchors
        anchor_indices = torch.randperm(num_samples)[:num_anchors]
    elif anchor_strategy == "fps":
        # Farthest Point Sampling
        anchor_indices = farthest_point_sampling(x_feats, num_anchors)
    elif anchor_strategy == "kmeans":
        # K-means clustering
        anchor_indices = kmeans_anchor_selection(x_feats, num_anchors)
    elif anchor_strategy == "topk":
        # Top-k selection (based on L2 norm or some other criterion)
        anchor_indices = topk_anchor_selection(x_feats, num_anchors)
    else:
        raise ValueError(f"Unknown anchor strategy: {anchor_strategy}")
    
    x_anchors = x_feats[anchor_indices]
    y_anchors = y_feats[anchor_indices]

    # Normalize the features for cosine similarity if needed
    if similarity_metric == "cosine":
        x_feats = F.normalize(x_feats, p=2, dim=-1)
        y_feats = F.normalize(y_feats, p=2, dim=-1)
        x_anchors = F.normalize(x_anchors, p=2, dim=-1)
        y_anchors = F.normalize(y_anchors, p=2, dim=-1)

    # Compute similarities between each sample and the anchors
    x_similarities = compute_similarity(x_feats, x_anchors, similarity_metric)  # N x num_anchors
    y_similarities = compute_similarity(y_feats, y_anchors, similarity_metric)  # N x num_anchors

    # KNN alignment: Use the nearest neighbors in both similarity spaces
    score = metrics.AlignmentMetrics.cycle_knn(x_similarities, y_similarities, topk=topk)
    
    return score


def compute_similarity(feats, anchors, similarity_metric="cosine"):
    """
    Computes similarity between features and anchors using the specified similarity metric.
    
    Args:
        feats: A torch tensor of shape N x L x D (features)
        anchors: A torch tensor of shape M x L x D (anchors)
        similarity_metric: The similarity metric to be used ("cosine", "euclidean")
    
    Returns:
        similarities: A similarity matrix of shape N x M
    """
    if similarity_metric == "cosine":
        # Compute cosine similarity
        similarities = torch.mm(feats, anchors.t())  # N x M
    elif similarity_metric == "euclidean":
        # Compute Euclidean distance and convert it to similarity
        dists = torch.cdist(feats, anchors, p=2)  # N x M
        similarities = -dists  # Higher distance -> lower similarity
    else:
        raise ValueError(f"Unknown similarity metric: {similarity_metric}")
    
    return similarities


def farthest_point_sampling(feats, num_anchors):
    """
    Implements Farthest Point Sampling (FPS) to select anchors.
    
    Args:
        feats: A torch tensor of shape N x D (features)
        num_anchors: Number of anchors to sample
    
    Returns:
        anchor_indices: The indices of selected anchors
    """
    feats = feats.to(device).numpy()  # Convert to numpy for easier distance computations
    num_samples = feats.shape[0]
    anchor_indices = [0]  # Start with the first point as anchor
    
    for _ in range(1, num_anchors):
        dist_matrix = cdist(feats, feats[anchor_indices], metric="euclidean")
        min_dists = dist_matrix.min(axis=1)
        next_anchor = min_dists.argmax()
        anchor_indices.append(next_anchor)
    
    return torch.tensor(anchor_indices)


def kmeans_anchor_selection(feats, num_anchors):
    """
    Implements K-means clustering to select anchors based on cluster centroids.
    
    Args:
        feats: A torch tensor of shape N x D (features)
        num_anchors: Number of anchors (clusters)
    
    Returns:
        anchor_indices: The indices of selected anchor points (cluster centroids)
    """
    feats = feats.to(device).numpy()  # Convert to numpy for KMeans
    kmeans = KMeans(n_clusters=num_anchors).fit(feats)
    centroids = kmeans.cluster_centers_
    
    # Find the closest points to the centroids
    dist_matrix = cdist(feats, centroids, metric="euclidean")
    anchor_indices = dist_matrix.argmin(axis=0)
    
    return torch.tensor(anchor_indices)


def topk_anchor_selection(feats, num_anchors):
    """
    Selects the top-k points based on their L2 norm.
    
    Args:
        feats: A torch tensor of shape N x D (features)
        num_anchors: Number of anchors to select
    
    Returns:
        anchor_indices: The indices of the top-k points
    """
    norms = feats.norm(p=2, dim=-1)
    anchor_indices = torch.topk(norms, num_anchors).indices
    
    return anchor_indices

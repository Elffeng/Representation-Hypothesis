import os 
import torch

import platonic
from datasets import load_dataset
from measure_alignment import compute_score, prepare_features


class Alignment():

    def __init__(self, dataset, subset, models=[], device="cuda", dtype=torch.bfloat16):
        
        if dataset != "minhuh/prh":
            # TODO: support external datasets in the future
            raise ValueError(f"dataset {dataset} not supported")
            
        if subset not in platonic.SUPPORTED_DATASETS:
            raise ValueError(f"subset {subset} not supported for dataset {dataset}")
        
        self.models = models
        self.device = device
        self.dtype = dtype

        # loads the features from path if it does not exist it will download
        self.features = {}
        for m in models:
            # feat_path = platonic.SUPPORTED_DATASETS[subset][m]["path"]
            # feat_url = platonic.SUPPORTED_DATASETS[subset][m]["url"]
            #
            # if not os.path.exists(feat_path):
            #     print(f"downloading features for {m} in {dataset}/{subset} from {feat_url}")
            #
            #     # download and save the features in the feat_path
            #     os.makedirs(os.path.dirname(feat_path), exist_ok=True)
            #     exit_code = os.system(f"wget {feat_url} -O {feat_path}")
            #     if exit_code != 0:
            #         raise ValueError(f"Failed to download features for {m} in {dataset}/{subset}")
            #
            #     if not os.path.exists(feat_path):
            #         raise ValueError(f"feature path {feat_path} does not exist for {m} in {dataset}/{subset}")
            feat_path = "D:\phd6\parttime/representation convergence\Representation-Hypothesis/results/features\minhuh\prh\wit_1024/bigscience_bloomz-560m_pool-avg.pt"
            self.features[m] = self.load_features(feat_path)
            
        # download dataset from huggingface        
        self.dataset = load_dataset(dataset, revision=subset, split='train', cache_dir='./wit_1024')
        self.dataset = self.dataset.select(range(12))
        return
    

    def load_features(self, feat_path):
        """ loads features for a model """
        return torch.load(feat_path, map_location=self.device)["feats"].to(dtype=self.dtype)

    
    def get_data(self, modality):
        """ load data 
        TODO: use multiprocessing to speed up loading
        
        """
        if modality == "text": # list of strings
            return [x['text'][0] for x in self.dataset]
        elif modality == "image": # list of PIL images
            return [x['image'] for x in self.dataset]
        else:
            raise ValueError(f"modality {modality} not supported")
    
    def score(self, features, metric, *args, **kwargs):
        """ 
        Args:
            features (torch.Tensor): features to compare
            metric (str): metric to use
            *args: additional arguments for compute_score / metrics.AlignmentMetrics
            **kwargs: additional keyword arguments for compute_score / metrics.AlignmentMetrics
        Returns:
            dict: scores for each model organized as 
                {model_name: (score, layer_indices)} 
                layer_indices are the index of the layer with maximal alignment
        """
        scores = {}
        for m in self.models:
            scores[m] = compute_score(
                prepare_features(features, exact=True).to(device=self.device, dtype=self.dtype),  # lvm的
                prepare_features(self.features[m], exact=True).to(device=self.device, dtype=self.dtype) , # llm的
                metric=metric,
                *args, 
                **kwargs
            )

        return scores
    
    
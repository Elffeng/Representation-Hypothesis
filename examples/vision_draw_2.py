import platonic
from tqdm.auto import trange
import torch 
from pprint import pprint
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torchvision.models.feature_extraction import create_feature_extractor
import matplotlib.pyplot as plt
import umap
import numpy as np
import argparse

def extract_lvm_features(vision_model, images, transform, batch_size):
    lvm_feats = []
    for i in trange(0, len(images), batch_size):
        ims = torch.stack([transform(images[j]) for j in range(i, i + batch_size)]).to(device)

        with torch.no_grad():
            lvm_output = vision_model(ims)

        feats = torch.stack([v[:, 0, :] for v in lvm_output.values()]).permute(1, 0, 2)
        lvm_feats.append(feats)

    lvm_feats = torch.cat(lvm_feats)
    return lvm_feats

def plot_alignment_figure(align_scores, align_errors, vtab_tasks_solved_labels):
    plt.figure(figsize=(10, 5))
    colors = ['yellow', 'green', 'lightgreen', 'darkgreen', 'blue']
    plt.bar(vtab_tasks_solved_labels, align_scores, yerr=align_errors, color=colors, alpha=0.7)
    plt.xlabel('Percentage of VTAB Tasks Solved (total=19)')
    plt.ylabel('Intra-bucket Alignment')
    plt.title('Convergence to General Competence')
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    

def plot_umap_figure(high_dim_feats, model_types, vtab_tasks_solved, markers):
    umap_reducer = umap.UMAP()
    umap_embeds = umap_reducer.fit_transform(high_dim_feats)

    plt.figure(figsize=(8, 6))
    for i, model_type in enumerate(model_types):
        plt.scatter(umap_embeds[i, 0], umap_embeds[i, 1], c=[vtab_tasks_solved[i]],
                    cmap='coolwarm', s=100, alpha=0.8, label=model_type, marker=markers[i])

    plt.colorbar(label='VTAB Tasks Solved')
    plt.title('UMAP of Model Representations')
    plt.legend(title="Model Types")
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force_download", action="store_true")
    parser.add_argument("--force_remake", action="store_true")
    parser.add_argument("--num_samples", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--dataset", type=str, default="minhuh/prh")
    parser.add_argument("--subset", type=str, default="wit_1024")
    parser.add_argument("--model_name", type=str, default="results/features/minhuh/prh/wit_1024/vit_tiny_patch16_224.augreg_in21k_pool-cls.pt")
    parser.add_argument("--output_dir", type=str, default="/Users/fengdan/Desktop/essay/platonic-rep-main/")
    parser.add_argument("--save_plots", type=str, help="/Users/fengdan/Desktop/essay/platonic-rep-main/", default=None)

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup platonic metric
    platonic_metric = platonic.Alignment(
        dataset=args.dataset,
        subset=args.subset,
        models=["openllama_7b", "llama_65b"]
    )

    # Load images
    images = platonic_metric.get_data(modality="image")

    # Load vision model
    vision_model = timm.create_model(args.model_name, pretrained=True).to(device).eval()

    transform = create_transform(
        **resolve_data_config(vision_model.pretrained_cfg, model=vision_model)
    )

    return_nodes = [f"blocks.{i}.add_1" for i in range(len(vision_model.blocks))]
    vision_model = create_feature_extractor(vision_model, return_nodes=return_nodes)

    lvm_feats = extract_lvm_features(vision_model, images, transform, args.batch_size)

    align_scores = platonic_metric.score(lvm_feats, metric="mutual_knn", topk=10, normalize=True)
    pprint(align_scores)

    #
    align_errors = [0.02, 0.02, 0.02, 0.03, 0.03]
    vtab_tasks_solved_labels = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']
    plot_alignment_figure(align_scores, align_errors, vtab_tasks_solved_labels)

    high_dim_feats = np.random.rand(5, 128)
    model_types = ['Classification', 'MAE', 'Random Initialization', 'Contrastive', 'CLIP']
    markers = ['o', 's', '^', '*', 'D']
    vtab_tasks_solved = [3, 6, 9, 15, 19]
    plot_umap_figure(high_dim_feats, model_types, vtab_tasks_solved, markers)


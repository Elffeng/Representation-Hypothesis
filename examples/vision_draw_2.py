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


# setup platonic metric
platonic_metric = platonic.Alignment(
                    dataset="minhuh/prh", # <--- this is the dataset 
                    subset="wit_1024",    # <--- this is the subset
                    models=["openllama_7b", "llama_65b"], 
                    ) # you can also pass in device and dtype as arguments

# load images
images = platonic_metric.get_data(modality="image")

# your model (e.g. we will use dinov2 as an example)
model_name = "vit_giant_patch14_dinov2.lvd142m"
vision_model = timm.create_model(model_name, pretrained=True).cuda().eval()

transform = create_transform(
    **resolve_data_config(vision_model.pretrained_cfg, model=vision_model)
)

# extract features
return_nodes = [f"blocks.{i}.add_1" for i in range(len(vision_model.blocks))]
vision_model = create_feature_extractor(vision_model, return_nodes=return_nodes)

lvm_feats = []
batch_size = 32


for i in trange(0, len(images), batch_size):
    ims = torch.stack([transform(images[j]) for j in range(i,i+batch_size)]).cuda()

    with torch.no_grad():
        lvm_output = vision_model(ims)

    feats = torch.stack([v[:, 0, :] for v in lvm_output.values()]).permute(1, 0, 2)
    lvm_feats.append(feats)
    
# compute score 
lvm_feats = torch.cat(lvm_feats)
score = platonic_metric.score(lvm_feats, metric="mutual_knn", topk=10, normalize=True)
pprint(score) # it will print the score and the index of the layer the maximal alignment happened

#画figure2
# align_scores = [0.05, 0.1, 0.15, 0.30, 0.40]
# align_errors = [0.02, 0.02, 0.02, 0.03, 0.03] #把measure_align的分数结果填到这里 
# vtab_tasks_solved_labels = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']

# # 绘制柱状图，包含误差条
# plt.figure(figsize=(10,5))
# colors = ['yellow', 'green', 'lightgreen', 'darkgreen', 'blue']  # 颜色渐变
# plt.bar(vtab_tasks_solved_labels, align_scores, yerr=align_errors, color=colors, alpha=0.7)
# plt.xlabel('Percentage of VTAB Tasks Solved (total=19)')
# plt.ylabel('Intra-bucket Alignment')
# plt.title('Convergence to General Competence')
# plt.show()

# # 2. 右图：UMAP 可视化
# high_dim_feats = np.random.rand(5, 128)  # 5个模型的128维特征
# model_types = ['Classification', 'MAE', 'Random Initialization', 'Contrastive', 'CLIP']  # 示例模型类型
# markers = ['o', 's', '^', '*', 'D']  # 不同模型类型的标记符号
# vtab_tasks_solved = [3, 6, 9, 15, 19]  # VTAB任务解决数量

# # UMAP降维
# umap_reducer = umap.UMAP()
# umap_embeds = umap_reducer.fit_transform(high_dim_feats)

# # UMAP散点图
# plt.figure(figsize=(8,6))
# for i, model_type in enumerate(model_types):
#     plt.scatter(umap_embeds[i, 0], umap_embeds[i, 1], c=[vtab_tasks_solved[i]], cmap='coolwarm', s=100, alpha=0.8, label=model_type, marker=markers[i])

# plt.colorbar(label='VTAB Tasks Solved')
# plt.title('UMAP of Model Representations')
# plt.legend(title="Model Types")
# plt.show()


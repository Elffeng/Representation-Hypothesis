import platonic
from models import load_llm, load_tokenizer
from tqdm.auto import trange
import torch
from pprint import pprint

# 设置语言模型和视觉模型
language_models = [
    "bigscience/bloomz-560m",
    "bigscience/bloomz-1b1",
    "openlm-research/open_llama_3b",
    "openlm-research/open_llama_7b",
]

vision_models = [
    "dinov2_small",
    "dinov2_base",
    "dinov2_large",
    "dinov2_giant"
]

# 初始化 platonic metric，使用图中的 dataset 和 subset
platonic_metric = platonic.Alignment(
    dataset="minhuh/prh",  
    subset="wit_1024",    
)

# 获取文本数据
texts = platonic_metric.get_data(modality="text")

# 循环遍历语言模型和视觉模型
for vision_model in vision_models:
    for model_name in language_models:
        print(f"Evaluating {model_name} with {vision_model}...")

        # 加载语言模型
        language_model = load_llm(model_name, qlora=False)
        device = next(language_model.parameters()).device
        tokenizer = load_tokenizer(model_name)

        # 提取特征
        tokens = tokenizer(texts, padding="longest", return_tensors="pt")
        llm_feats = []
        batch_size = 16

        for i in trange(0, len(texts), batch_size):
            token_inputs = {k: v[i:i+batch_size].to(device).long() for (k, v) in tokens.items()}
            with torch.no_grad():
                llm_output = language_model(
                    input_ids=token_inputs["input_ids"],
                    attention_mask=token_inputs["attention_mask"],
                )
            feats = torch.stack(llm_output["hidden_states"]).permute(1, 0, 2, 3).cpu()
            mask = token_inputs["attention_mask"].unsqueeze(-1).unsqueeze(1).cpu()
            feats = (feats * mask).sum(2) / mask.sum(2)
            llm_feats.append(feats)

        llm_feats = torch.cat(llm_feats)

        # 计算语言模型与视觉模型的对齐分数
        platonic_metric.models = [vision_model]
        score = platonic_metric.score(llm_feats, metric="mutual_knn", topk=10, normalize=True)
        
        # 打印模型与视觉模型的得分
        pprint(score)

import matplotlib.pyplot as plt

language_scores = [0.1, 0.2, 0.3, 0.4]  # 模拟语言模型性能数据
alignment_scores = [0.12, 0.14, 0.16, 0.18]  # 模拟对齐分数

plt.scatter(language_scores, alignment_scores)
plt.plot(language_scores, alignment_scores, label="Alignment to DinoV2")

plt.title("LANGUAGE and VISION models align")
plt.xlabel("LANGUAGE performance")
plt.ylabel("Alignment to DINOv2")
plt.legend()

plt.show()



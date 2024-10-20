import gc
import os
import argparse

from tqdm import trange
import torch

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torchvision.models.feature_extraction import create_feature_extractor

from datasets import load_dataset
from tasks import get_models
from models import load_llm, load_tokenizer
import utils 
    

def extract_llm_features(filenames, dataset, args):
    """
    Extracts features from language models.
    Args:
        filenames: list of language model names by huggingface identifiers
        dataset: huggingface dataset
        args: argparse arguments
    """
    dataset = dataset.select(range(12))
    texts = [str(x['text'][args.caption_idx]) for x in dataset]
        
    for llm_model_name in filenames[::-1]:
        save_path = utils.to_feature_filename(
            args.output_dir, args.dataset, args.subset, llm_model_name,
            pool=args.pool, prompt=args.prompt, caption_idx=args.caption_idx,
        )
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        print(f"\ndataset: \t{args.dataset}")
        print(f"subset:    \t{args.subset}")
        print(f"processing:\t{llm_model_name}")
        print(f'save_path: \t{save_path}')
        
        if os.path.exists(save_path) and not args.force_remake:
            print("file exists. skipping")
            continue
        #CPU
        language_model = load_llm(llm_model_name, qlora=args.qlora, force_download=args.force_download)
        llm_param_count = sum([p.numel() for p in language_model.parameters()])
        tokenizer = load_tokenizer(llm_model_name)
    
        tokens = tokenizer(texts, padding="longest", return_tensors="pt")        
        llm_feats, losses, bpb_losses = [], [], []

        for i in trange(0, len(dataset), args.batch_size):
            token_inputs = {k: v[i:i+args.batch_size].to(device).long() for (k, v) in tokens.items()}

            with torch.no_grad():
                llm_output = language_model(
                    input_ids=token_inputs["input_ids"],
                    attention_mask=token_inputs["attention_mask"],
                )

                loss, avg_loss = utils.cross_entropy_loss(token_inputs, llm_output)
                losses.extend(avg_loss.to(device))

                bpb = utils.cross_entropy_to_bits_per_unit(loss.to(device), texts[i:i+args.batch_size], unit="byte")
                bpb_losses.extend(bpb)
                
                if args.pool == 'avg':
                    hidden_states = torch.stack(llm_output["hidden_states"])  # (num_layers, batch_size, seq_len, hidden_dim)
                    mask = token_inputs["attention_mask"].unsqueeze(-1).unsqueeze(1)  # (batch_size, 1, seq_len, 1)
                    masked_hidden_states = hidden_states * mask
                    avg_feats = masked_hidden_states.sum(dim=2) / mask.sum(dim=2)
                    avg_feats = avg_feats.mean(dim=0)
                    
                elif args.pool == 'last':
                    feats = [v[:, -1, :] for v in llm_output["hidden_states"]]
                    feats = torch.stack(feats).permute(1, 0, 2)
                llm_feats.append(feats.to(device))

        print(f"average loss:\t{torch.stack(losses).mean().item()}")
        save_dict = {
            "feats": torch.cat(llm_feats).to(device),
            "num_params": llm_param_count,
            "mask": tokens["attention_mask"].to(device),
            "loss": torch.stack(losses).mean(),
            "bpb": torch.stack(bpb_losses).mean(),
        }

        torch.save(save_dict, save_path)
        del language_model, tokenizer, llm_feats, llm_output
        gc.collect()
    return
    
        
def extract_lvm_features(filenames, dataset, args):
    """
    Extracts features from vision models.
    Args:
        filenames: list of vision model names by timm identifiers
        image_file_paths: list of image file paths
        args: argparse arguments
    """
    assert args.pool == 'cls', "pooling is not supported for lvm features"
    
    for lvm_model_name in filenames:
        assert 'vit' in lvm_model_name, "only vision transformers are supported"
        
        save_path = utils.to_feature_filename(
            args.output_dir, args.dataset, args.subset, lvm_model_name,
            pool=args.pool, prompt=None, caption_idx=None,
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        print(f"\ndataset: \t{args.dataset}")
        print(f"subset:    \t{args.subset}")
        print(f"processing:\t{lvm_model_name}")
        print(f'save_path: \t{save_path}')

        # if os.path.exists(save_path) and not args.force_remake:
        #     print("file exists. skipping")
        #     continue
        vision_model = timm.create_model(lvm_model_name, pretrained=True).eval()
        vision_model = vision_model.to(device)
        lvm_param_count = sum([p.numel() for p in vision_model.parameters()])

        transform = create_transform(
            **resolve_data_config(vision_model.pretrained_cfg, model=vision_model)
        )

        if "vit" in lvm_model_name:
            return_nodes = [f"blocks.{i}.add_1" for i in range(len(vision_model.blocks))]
        else:
            raise NotImplementedError(f"unknown model {lvm_model_name}")

        vision_model = create_feature_extractor(vision_model, return_nodes=return_nodes)
        lvm_feats = []

        for i in trange(0, len(dataset), args.batch_size):
            with torch.no_grad():
                ims = torch.stack([transform(dataset[j]['image']) for j in range(i, i+args.batch_size)]).to(device)
                lvm_output = vision_model(ims)

                if args.pool == "cls":
                    feats = [v[:, 0, :] for v in lvm_output.values()]
                    feats = torch.stack(feats).permute(1, 0, 2)
                    
                lvm_feats.append(feats.to(device))

        torch.save({"feats": torch.cat(lvm_feats), "num_params": lvm_param_count}, save_path)

        del vision_model, transform, lvm_feats, lvm_output
        # torch.cuda.empty_cache()
        # torch.cuda.ipc_collect()
        gc.collect()



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--force_download", action="store_true")
    parser.add_argument("--force_remake",   action="store_true")
    parser.add_argument("--num_samples",    type=int, default=1024)
    parser.add_argument("--batch_size",     type=int, default=4)
    parser.add_argument("--pool",           type=str, default='avg', choices=['avg', 'cls'])
    parser.add_argument("--prompt",         action="store_true")
    parser.add_argument("--dataset",        type=str, default="prh")
    parser.add_argument("--subset",         type=str, default="wit_1024")
    parser.add_argument("--caption_idx",    type=int, default=0)
    parser.add_argument("--modelset",       type=str, default="val", choices=["val", "test"])
    parser.add_argument("--modality",       type=str, default="all", choices=["vision", "language", "all"])
    parser.add_argument("--output_dir",     type=str, default="/Users/fengdan/Desktop/essay/platonic-rep-main/")
    parser.add_argument("--qlora",          action="store_true")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    if args.qlora:
        print(f"QLoRA is set to True. The alignment score will be slightly off.")

    llm_models, lvm_models = get_models(args.modelset, modality=args.modality)
    
    # load dataset once outside    
    dataset = load_dataset(args.dataset, revision=args.subset, split='train')

    if args.modality in ["all", "language"]:
        # extract all language model features
        extract_llm_features(llm_models, dataset, args)
    
    if args.modality in ["all", "vision"]:
        # extract all vision model features
        extract_lvm_features(lvm_models, dataset, args)


"""
主要功能：从语言模型（LLM）和视觉模型（LVM）中提取特征，并将它们保存在指定的路径下
整体逻辑：输入的数据集以及模型列表，计算并保存这些模型的特征向量
"""

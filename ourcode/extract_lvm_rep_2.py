import argparse
import os
import numpy as np
import timm
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, AutoConfig
from torchvision.models.feature_extraction import create_feature_extractor
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from sklearn.cross_decomposition import CCA

import metrics
import utils


def extract_and_save_lvm_rep(args, images, lvm_model, tokenizer, ):
    if "vit" in args.lvm_model_name:
        return_nodes = [f"blocks.{i}.add_1" for i in range(len(lvm_model.blocks))]
    else:
        raise NotImplementedError(f"unknown model {args.lvm_model_name}")
    lvm_model = create_feature_extractor(lvm_model, return_nodes=return_nodes)

    lvm_feats = []

    with torch.no_grad():
        for image in images:
            input_tokens = tokenizer(image).unsqueeze(0)
            lvm_output = lvm_model(input_tokens)

            if args.pool == "cls":
                feats = [v[:, 0, :] for v in lvm_output.values()]
                feats = torch.stack(feats).permute(1, 0, 2)
            lvm_feats.append(feats)

    save_dict = {
        "feats": torch.cat(lvm_feats),
    }
    save_path = utils.to_feature_filename(
        args.output_dir, args.dataset, args.lvm_model_name,
        pool=args.pool,
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(save_dict, save_path)


def check_bfloat16_support():
    """ checks if cuda driver/device supports bfloat16 computation """
    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        compute_capability = torch.cuda.get_device_capability(current_device)
        if compute_capability[0] >= 7:  # Check if device supports bfloat16
            return True
        else:
            return False
    else:
        return None


def auto_determine_dtype():
    """ automatic dtype setting. override this if you want to force a specific dtype """
    compute_dtype = torch.bfloat16 if check_bfloat16_support() else torch.float32
    torch_dtype = torch.bfloat16 if check_bfloat16_support() else torch.float32
    print(f"compute_dtype:\t{compute_dtype}")
    print(f"torch_dtype:\t{torch_dtype}")
    return compute_dtype, torch_dtype


def load_model(lvm_model_name):
    vision_model = timm.create_model(lvm_model_name, pretrained=True)
    lvm_param_count = sum([p.numel() for p in vision_model.parameters()])

    transform = create_transform(
        **resolve_data_config(vision_model.pretrained_cfg, model=vision_model)
    )

    return vision_model, transform


def load_tokenizer(llm_model_name):
    tokenizer = AutoTokenizer.from_pretrained(llm_model_name)

    if "huggyllama" in llm_model_name:
        tokenizer.pad_token = "[PAD]"
    else:
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    tokenizer.padding_side = "left"
    return tokenizer


def load_images(args):
    from utils import load_wit_1024, load_multi30k, load_flowers102
    if args.dataset == "wit_1024":
        images = load_wit_1024(args, modal='image')
    elif args.dataset == "multi30k":
        images = load_multi30k(args, modal='image')
    elif args.dataset == "flowers102":
        images = load_flowers102(modal='image')
    return images


def load_representation(save_path, q=0.95, exact=False):
    hidden_states = torch.load(save_path)
    # feats = metrics.remove_outliers(hidden_states.float(), q=q, exact=exact)
    return hidden_states


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force_download", action="store_true")
    parser.add_argument("--force_remake", action="store_true")
    parser.add_argument("--num_samples", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--pool", type=str, default='cls', choices=['cls'])
    parser.add_argument("--dataset", type=str, default="flowers102", choices=['wit_1024', 'multi30k', 'flowers102'])
    parser.add_argument("--lvm_model_name", type=str, default="vit_tiny_patch16_224.augreg_in21k",
                        choices=[""])
    parser.add_argument("--output_dir", type=str, default="../results/features")
    parser.add_argument("--qlora", action="store_true")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    return args


if __name__ == "__main__":
    args = get_args()
    images = load_images(args)
    lvm_model, transform = load_model(args.lvm_model_name)
    extract_and_save_lvm_rep(args=args, images=images, lvm_model=lvm_model, tokenizer=transform)

import argparse

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


def extract_and_save_lvm_rep(args, images, lvm_model, tokenizer,):
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
        args.output_dir, args.dataset, args.subset, args.lvm_model_name,
        pool=args.pool, prompt=None, caption_idx=None,
    )
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
    dataset = load_dataset(args.dataset, revision=args.subset, split='train', cache_dir='./wit_1024')
    dataset = dataset.select(range(12))
    images = [x['image'] for x in dataset]
    return images



def load_representation(save_path, q=0.95, exact=False):
    """
        准备特征，通过去除异常值和标准化特征
    """
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
    # parser.add_argument("--prompt", action="store_true")
    parser.add_argument("--dataset", type=str, default="minhuh/prh")
    parser.add_argument("--subset", type=str, default="wit_1024")
    parser.add_argument("--caption_idx", type=int, default=0)
    parser.add_argument("--modelset", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--lvm_model_name", type=str, default="vit_tiny_patch16_224.augreg_in21k", choices=["val", "test"])
    parser.add_argument("--modality", type=str, default="vision", choices=["vision", "language", "all"])
    parser.add_argument("--output_dir", type=str, default="./results/features")
    parser.add_argument("--qlora", action="store_true")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    return args


if __name__=="__main__":
    args = get_args()
    images = load_images(args)
    lvm_model, transform = load_model(args.lvm_model_name)
    extract_and_save_lvm_rep(args=args, images=images, lvm_model=lvm_model, tokenizer=transform)
    # load_representation("./results/features\minhuh\prh\wit_1024/bigscience_bloomz-560m_pool-avg.pt")

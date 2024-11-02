import argparse

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, AutoConfig
from sklearn.cross_decomposition import CCA
import os
import metrics
import utils


def extract_and_save_llm_rep(args, texts_en, texts_de, llm_model, tokenizer):
    llm_feats_en, llm_feats_de = [], []
    for text in texts_en:
        inputs = tokenizer(text, padding="longest", return_tensors="pt")
        input_ids = inputs["input_ids"]
        mask = inputs["attention_mask"].unsqueeze(-1).unsqueeze(1)

        with torch.no_grad():
            llm_output = llm_model(input_ids=input_ids)
            all_hidden_states = llm_output["hidden_states"]
            if args.pool == 'avg':
                feats = torch.stack(all_hidden_states).permute(1, 0, 2, 3)
                feats = (feats * mask).sum(2) / mask.sum(2)
            elif args.pool == 'last':
                feats = [v[:, -1, :] for v in all_hidden_states]
                feats = torch.stack(feats).permute(1, 0, 2)
            else:
                raise NotImplementedError(f"unknown pooling {args.pool}")
            llm_feats_en.append(feats.cpu())

    for text in texts_de:
        inputs = tokenizer(text, padding="longest", return_tensors="pt")
        input_ids = inputs["input_ids"]
        mask = inputs["attention_mask"].unsqueeze(-1).unsqueeze(1)

        with torch.no_grad():
            llm_output = llm_model(input_ids=input_ids)
            all_hidden_states = llm_output["hidden_states"]
            if args.pool == 'avg':
                feats = torch.stack(all_hidden_states).permute(1, 0, 2, 3)
                feats = (feats * mask).sum(2) / mask.sum(2)
            elif args.pool == 'last':
                feats = [v[:, -1, :] for v in all_hidden_states]
                feats = torch.stack(feats).permute(1, 0, 2)
            else:
                raise NotImplementedError(f"unknown pooling {args.pool}")
            llm_feats_de.append(feats.cpu())

    save_dict_en = {
    "feats_en": torch.cat(llm_feats_en),
}

    save_path_en = utils.to_feature_filename(
    args.output_dir, args.dataset, args.subset, args.llm_model_name,
    pool=args.pool, caption_idx=args.caption_idx, language="en"
)
    os.makedirs(os.path.dirname(save_path_en), exist_ok=True)
    torch.save(save_dict_en, save_path_en)

    save_dict_de = {
    "feats_de": torch.cat(llm_feats_de),
}
    save_path_de = utils.to_feature_filename(
    args.output_dir, args.dataset, args.subset, args.llm_model_name,
    pool=args.pool, caption_idx=args.caption_idx, language="de"
)
    os.makedirs(os.path.dirname(save_path_de), exist_ok=True)
    torch.save(save_dict_de, save_path_de)



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


def load_llm(llm_model_path, qlora=False, force_download=False, from_init=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_map = "auto" if device == "cuda" else None

    quantization_config = None
    if qlora:
        compute_dtype, torch_dtype = auto_determine_dtype()
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    if from_init:
        config = AutoConfig.from_pretrained(llm_model_path,
                                            device_map=device_map,
                                            quantization_config=quantization_config,
                                            torch_dtype=torch_dtype,
                                            force_download=force_download,
                                            output_hidden_states=True, )
        language_model = AutoModelForCausalLM.from_config(config)
    else:
        language_model = AutoModelForCausalLM.from_pretrained(
            llm_model_path,
            device_map=device_map,
            quantization_config=quantization_config,
            # torch_dtype=torch_dtype,
            force_download=force_download,
            output_hidden_states=True,
        )

    return language_model


def load_tokenizer(llm_model_name):
    tokenizer = AutoTokenizer.from_pretrained(llm_model_name)

    if "huggyllama" in llm_model_name:
        tokenizer.pad_token = "[PAD]"
    else:
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    tokenizer.padding_side = "left"
    return tokenizer





def load_text(args):
    dataset = load_dataset(args.dataset, split='train', cache_dir='./wit_1024')
    # dataset = dataset.select(range(12))
    dataset = dataset.select(range(50))
    english_texts = [str(x['en']) for x in dataset]
    german_texts = [str(x['de']) for x in dataset]
    # texts = [str(x['text'][args.caption_idx]) for x in dataset]
    return english_texts, german_texts


def load_model(args):
    language_model = load_llm(llm_model_path=args.llm_model_name, qlora=args.qlora, force_download=args.force_download)
    llm_param_count = sum([p.numel() for p in language_model.parameters()])
    tokenizer = load_tokenizer(llm_model_name=args.llm_model_name)
    return language_model, tokenizer



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
    parser.add_argument("--pool", type=str, default='avg', choices=['avg', 'cls'])
    # parser.add_argument("--prompt", action="store_true")
    parser.add_argument("--dataset", type=str, default="minhuh/prh")
    parser.add_argument("--subset", type=str, default="wit_1024")
    parser.add_argument("--caption_idx", type=int, default=0)
    parser.add_argument("--modelset", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--llm_model_name", type=str, default="bigscience/bloomz-560m", choices=["val", "test"])
    parser.add_argument("--modality", type=str, default="language", choices=["vision", "language", "all"])
    parser.add_argument("--output_dir", type=str, default="./results/features")
    parser.add_argument("--qlora", action="store_true")
    
    # 可以选择语言
    parser.add_argument("--language", type=str, default="en", choices=["en", "de"])

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    return args


if __name__=="__main__":
    args = get_args()
    texts_en, texts_de = load_text(args)
    llm_model, tokenizer = load_model(args)
    extract_and_save_llm_rep(args=args, texts_en=texts_en, texts_de=texts_de, llm_model=llm_model, tokenizer=tokenizer)
    # load_representation("D:\phd6\parttime/representation convergence\Representation-Hypothesis/results/features\minhuh\prh\wit_1024/bigscience_bloomz-560m_pool-avg.pt")
    load_representation("/Users/fengdan/Desktop/essay/platonic-rep-main/results/features/minhuh/prh/wit_1024/vit_tiny_patch16_224.augreg_in21k_pool-cls.pt")

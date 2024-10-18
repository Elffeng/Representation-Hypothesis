import argparse

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, AutoConfig
from sklearn.cross_decomposition import CCA
import utils


def extract_and_save_llm_rep(args, texts, llm_model, tokenizer,):

    llm_feats, losses, bpb_losses = [], [], []

    for text in texts:
        inputs = tokenizer(text, padding="longest", return_tensors="pt")
        input_ids = inputs["input_ids"]
        mask = inputs["attention_mask"].unsqueeze(-1).unsqueeze(1)  # (batch_size, 1, seq_len, 1)

        with torch.no_grad():
            llm_output = llm_model(input_ids=input_ids,)
            all_hidden_states = llm_output["hidden_states"] # (num_layers, batch_size, seq_len, hidden_dim)
            if args.pool == 'avg':
                feats = torch.stack(llm_output["hidden_states"]).permute(1, 0, 2, 3)
                feats = (feats * mask).sum(2) / mask.sum(2)
            elif args.pool == 'last':
                feats = [v[:, -1, :] for v in llm_output["hidden_states"]]
                feats = torch.stack(feats).permute(1, 0, 2)
            else:
                raise NotImplementedError(f"unknown pooling {args.pool}")
            llm_feats.append(feats.cpu())

            # loss, avg_loss = utils.cross_entropy_loss(token_inputs, llm_output)
            # losses.extend(avg_loss.cpu())
            #
            # bpb = utils.cross_entropy_to_bits_per_unit(loss.cpu(), texts[i:i + args.batch_size], unit="byte")
            # bpb_losses.extend(bpb)
        # print(f"average loss:\t{torch.stack(losses).mean().item()}")
        save_dict = {
            "feats": torch.cat(llm_feats),
            # "num_params": llm_param_count,
            # "mask": tokens["attention_mask"],
            # "loss": torch.stack(losses).mean(),
            # "bpb": torch.stack(bpb_losses).mean(),
        }
        save_path = utils.to_feature_filename(
            args.output_dir, args.dataset, args.subset, args.llm_model_name,
            pool=args.pool, prompt=args.prompt, caption_idx=args.caption_idx,
        )
        torch.save(save_dict, save_path)
        # del llm_model, tokenizer, llm_feats, llm_output
        # torch.cuda.empty_cache()
        # torch.cuda.ipc_collect()
        # gc.collect()



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
            torch_dtype=torch_dtype,
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
    dataset = load_dataset(args.dataset, revision=args.subset, split='train', cache_dir='./wit_1024')
    dataset = dataset.select(range(12))
    texts = [str(x['text'][args.caption_idx]) for x in dataset]
    return texts


def load_model(args):
    language_model = load_llm(llm_model_path=args.llm_model_name, qlora=args.qlora, force_download=args.force_download)
    llm_param_count = sum([p.numel() for p in language_model.parameters()])
    tokenizer = load_tokenizer(llm_model_name=args.llm_model_name)
    return language_model, tokenizer


def load_metrics(args):
    SUPPORTED_METRICS = [
        "cycle_knn",
        "mutual_knn",
        "lcs_knn",
        "cka",
        "unbiased_cka",
        "cknna",
        "svcca",
        "edit_distance_knn",
    ]

    def compute_nearest_neighbors(feats, topk=1):
        """
        Compute the nearest neighbors of feats
        Args:
            feats: a torch tensor of shape N x D
            topk: the number of nearest neighbors to return
        Returns:
            knn: a torch tensor of shape N x topk
        """
        assert feats.ndim == 2, f"Expected feats to be 2D, got {feats.ndim}"
        knn = (
            (feats @ feats.T).fill_diagonal_(-1e8).argsort(dim=1, descending=True)[:, :topk]
        )
        return knn

    def compute_knn_accuracy(knn):
        """
        Compute the accuracy of the nearest neighbors. Assumes index is the gt label.
        Args:
            knn: a torch tensor of shape N x topk
        Returns:
            acc: a float representing the accuracy
        """
        n = knn.shape[0]
        acc = knn == torch.arange(n, device=knn.device).view(-1, 1, 1)
        acc = acc.float().view(n, -1).max(dim=1).values.mean()
        return acc

    def longest_ordinal_sequence(X, Y):
        """ For each pair in X and Y, compute the length of the longest sub-sequence (LCS) """

        def compute_distance(X, Y, dist_fn):
            """ compute distance in parallel"""
            B, N = X.shape
            distances = np.zeros(B)
            X, Y = X.cpu().numpy(), Y.cpu().numpy()

            if pymp_available:
                with pymp.Parallel(4) as p:
                    for i in p.range(B):
                        distances[i] = dist_fn(X[i], Y[i])
            else:
                for i in range(B):
                    distances[i] = dist_fn(X[i], Y[i])
            return torch.tensor(distances)

        def lcs_length(x, y):
            """
            Compute the length of the longest common subsequence between two sequences.
            This is a classic dynamic programming implementation.
            """
            m, n = len(x), len(y)
            dp = [[0] * (n + 1) for _ in range(m + 1)]

            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if x[i - 1] == y[j - 1]:
                        dp[i][j] = dp[i - 1][j - 1] + 1
                    else:
                        dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
            return dp[m][n]

        lcs = compute_distance(X, Y, lcs_length)
        return lcs

    def cycle_knn(feats_A, feats_B, topk):
        """
        LLM nearest neighbors -> Query Language Pair -> LVM nearest neighbors
        Args:
            feats_A: A torch tensor of shape N x feat_dim
            feats_B: A torch tensor of shape N x feat_dim

        Returns:
            acc: a float representing the accuracy
        """
        knn_A = compute_nearest_neighbors(feats_A, topk)
        knn_B = compute_nearest_neighbors(feats_B, topk)
        return compute_knn_accuracy(knn_A[knn_B]).item()

    def mutual_knn(feats_A, feats_B, topk):
        """
        Computes the mutual KNN accuracy.

        Args:
            feats_A: A torch tensor of shape N x feat_dim
            feats_B: A torch tensor of shape N x feat_dim

        Returns:
            A float representing the mutual KNN accuracy
        """
        knn_A = compute_nearest_neighbors(feats_A, topk)
        knn_B = compute_nearest_neighbors(feats_B, topk)

        n = knn_A.shape[0]
        topk = knn_A.shape[1]

        # Create a range tensor for indexing
        range_tensor = torch.arange(n, device=knn_A.device).unsqueeze(1)

        # Create binary masks for knn_A and knn_B
        lvm_mask = torch.zeros(n, n, device=knn_A.device)
        llm_mask = torch.zeros(n, n, device=knn_A.device)

        lvm_mask[range_tensor, knn_A] = 1.0
        llm_mask[range_tensor, knn_B] = 1.0

        acc = (lvm_mask * llm_mask).sum(dim=1) / topk

        return acc.mean().item()

    def lcs_knn(feats_A, feats_B, topk):
        knn_A = compute_nearest_neighbors(feats_A, topk)
        knn_B = compute_nearest_neighbors(feats_B, topk)
        score = longest_ordinal_sequence(knn_A, knn_B).float().mean()
        return score

    @staticmethod
    def cka(feats_A, feats_B, kernel_metric='ip', rbf_sigma=1.0, unbiased=False):
        """Computes the unbiased Centered Kernel Alignment (CKA) between features."""

        if kernel_metric == 'ip':
            # Compute kernel matrices for the linear case
            K = torch.mm(feats_A, feats_A.T)
            L = torch.mm(feats_B, feats_B.T)
        elif kernel_metric == 'rbf':
            # COMPUTES RBF KERNEL
            K = torch.exp(-torch.cdist(feats_A, feats_A) ** 2 / (2 * rbf_sigma ** 2))
            L = torch.exp(-torch.cdist(feats_B, feats_B) ** 2 / (2 * rbf_sigma ** 2))
        else:
            raise ValueError(f"Invalid kernel metric {kernel_metric}")

        # Compute HSIC values
        hsic_fn = hsic_unbiased if unbiased else hsic_biased
        hsic_kk = hsic_fn(K, K)
        hsic_ll = hsic_fn(L, L)
        hsic_kl = hsic_fn(K, L)

        # Compute CKA
        # print('hsic', hsic_kl)
        cka_value = hsic_kl / (torch.sqrt(hsic_kk * hsic_ll) + 1e-6)
        return cka_value.item()

    @staticmethod
    def unbiased_cka(*args, **kwargs):
        kwargs['unbiased'] = True
        return AlignmentMetrics.cka(*args, **kwargs)

    @staticmethod
    def svcca(feats_A, feats_B, cca_dim=10):

        # Center and scale the activations
        def preprocess_activations(act):
            act = act - torch.mean(act, axis=0)
            act = act / (torch.std(act, axis=0) + 1e-8)
            return act

        feats_A = preprocess_activations(feats_A)
        feats_B = preprocess_activations(feats_B)

        # Compute SVD
        U1, _, _ = torch.svd_lowrank(feats_A, q=cca_dim)
        U2, _, _ = torch.svd_lowrank(feats_B, q=cca_dim)

        U1 = U1.cpu().detach().numpy()
        U2 = U2.cpu().detach().numpy()

        # Compute CCA
        cca = CCA(n_components=cca_dim)
        cca.fit(U1, U2)
        U1_c, U2_c = cca.transform(U1, U2)

        # sometimes it goes to nan, this is just to avoid that
        U1_c += 1e-10 * np.random.randn(*U1_c.shape)
        U2_c += 1e-10 * np.random.randn(*U2_c.shape)

        # Compute SVCCA similarity
        svcca_similarity = np.mean(
            [np.corrcoef(U1_c[:, i], U2_c[:, i])[0, 1] for i in range(cca_dim)]
        )
        return svcca_similarity

    @staticmethod
    def edit_distance_knn(feats_A, feats_B, topk):
        """
        Computes the edit distance between the nearest neighbors of feats_A and feats_B.
        """
        knn_A = compute_nearest_neighbors(feats_A, topk)
        knn_B = compute_nearest_neighbors(feats_B, topk)

        # given N x topk with integer entries, compute edit distance
        n = knn_A.shape[0]
        topk = knn_A.shape[1]

        edit_distance = compute_distance(knn_A, knn_B, TAF.edit_distance)
        return 1 - torch.mean(edit_distance) / topk

    @staticmethod
    def cknna(feats_A, feats_B, topk=None, distance_agnostic=False, unbiased=True):
        """ similarity only cka variant """
        n = feats_A.shape[0]

        if topk < 2:
            raise ValueError("CKNNA requires topk >= 2")

        if topk is None:
            topk = feats_A.shape[0] - 1

        K = feats_A @ feats_A.T
        L = feats_B @ feats_B.T
        device = feats_A.device

        def similarity(K, L, topk):
            if unbiased:
                K_hat = K.clone().fill_diagonal_(float("-inf"))
                L_hat = L.clone().fill_diagonal_(float("-inf"))
            else:
                K_hat, L_hat = K, L

            # get topk indices for each row
            # if unbiased we cannot attend to the diagonal unless full topk
            # else we can attend to the diagonal
            _, topk_K_indices = torch.topk(K_hat, topk, dim=1)
            _, topk_L_indices = torch.topk(L_hat, topk, dim=1)

            # create masks for nearest neighbors
            mask_K = torch.zeros(n, n, device=device).scatter_(1, topk_K_indices, 1)
            mask_L = torch.zeros(n, n, device=device).scatter_(1, topk_L_indices, 1)

            # intersection of nearest neighbors
            mask = mask_K * mask_L

            if distance_agnostic:
                sim = mask * 1.0
            else:
                if unbiased:
                    sim = hsic_unbiased(mask * K, mask * L)
                else:
                    sim = hsic_biased(mask * K, mask * L)
            return sim

        sim_kl = similarity(K, L, topk)
        sim_kk = similarity(K, K, topk)
        sim_ll = similarity(L, L, topk)

        return sim_kl.item() / (torch.sqrt(sim_kk * sim_ll) + 1e-6).item()



def plot():
    pass


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force_download", action="store_true")
    parser.add_argument("--force_remake", action="store_true")
    parser.add_argument("--num_samples", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--pool", type=str, default='avg', choices=['avg', 'cls'])
    parser.add_argument("--prompt", action="store_true")
    parser.add_argument("--dataset", type=str, default="minhuh/prh")
    parser.add_argument("--subset", type=str, default="wit_1024")
    parser.add_argument("--caption_idx", type=int, default=0)
    parser.add_argument("--modelset", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--llm_model_name", type=str, default="bigscience/bloomz-560m", choices=["val", "test"])
    parser.add_argument("--modality", type=str, default="language", choices=["vision", "language", "all"])
    parser.add_argument("--output_dir", type=str, default="./results/features")
    parser.add_argument("--qlora", action="store_true")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    return args


if __name__=="__main__":
    args = get_args()
    texts = load_text(args)
    llm_model, tokenizer = load_model(args)
    extract_and_save_llm_rep(args=args, texts=texts, llm_model=llm_model, tokenizer=tokenizer)

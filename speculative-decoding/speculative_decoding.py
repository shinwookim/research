import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.nn import functional as F
from colorama import Fore, Style


def sample(logits: torch.Tensor, num_samples: int = 1):
    idx = torch.multinomial(logits, num_samples=num_samples)
    assert idx.item() != 0, "sampled token is <eos>"
    return idx


# Based on https://github.com/LeeSinLiang/microGPT (MIT License)
def top_k_top_p_filter(logits: torch.Tensor, top_k: int = 0, top_p: float = 0.0):
    """

    Args:
        logits (torch.Tensorpe_): 2D tensor with shape (batch, vocab)
        top_k (int, optional): top_k. Defaults to 0.
        top_p (float, optional): top_p. Defaults to 0.0.

    Returns:
        torch.Tensor: a renormalized logits
    """
    if top_k > 0:
        filter = torch.topk(logits, min(top_k, logits.size(-1)))[0]
        logits[logits < filter[:, [-1]]] = float("-inf")
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        filter = cumulative_probs > top_p
        filter[..., 1:] = filter[..., :-1].clone()
        filter[..., 0] = 0
        indices_to_remove = filter.scatter(1, sorted_indices, filter)
        logits[indices_to_remove] = float("-inf")
    return logits


def norm_logits(
    logits: torch.Tensor, temperature: float, top_k: int, top_p: float
) -> torch.Tensor:
    """

    Args:
        logits (torch.Tensor): shape (1, vocab)
        temperature (float): temperature
        top_k (float): top_k
        top_p (float): top_p

    Returns:
        torch.Tensor: next token with shape as (batch,  1)
    """
    assert logits.dim() == 2
    logits = logits / temperature
    logits = top_k_top_p_filter(logits, top_k=top_k, top_p=top_p)
    probs = F.softmax(logits, dim=1)
    return probs


def sample(probs: torch.Tensor, num_samples: int = 1):
    idx_next = torch.multinomial(probs, num_samples=num_samples)
    if idx_next.item() == 0:
        raise RuntimeError
    return idx_next


def max_fn(x):
    """
    norm(max (x, 0))
    """
    x_max = torch.where(x > 0, x, torch.zeros_like(x))
    x_max_sum = torch.sum(x_max, dim=1, keepdim=True)
    return x_max / x_max_sum


@torch.no_grad()
def speculative_decoding(
    prefix: torch.Tensor,
    approx_model: torch.nn.Module,
    target_model: torch.nn.Module,
    max_len: int,
    decoder,
    gamma: int = 4,
    temperature: float = 1,
    top_k: int = 0,
    top_p: float = 0,
) -> torch.Tensor:
    """
    DeepMind version Speculative Sampling.
    Accelerating Large Language Model Decoding with Speculative Sampling
    https://arxiv.org/abs/2302.01318
    No KV Cache Optimization

    Args:
        x (torch.Tensor): input sequence, (batch, prefix_seqlen), Note that the batch dim is always 1 now.
        approx_model (torch.nn.Module): approx model, the small one
        target_model (torch.nn.Module): target model, the large one
        max_len (int): the max overall generated tokens number.
        gamma (int): $\gamma$, the token number small model guesses.
        temperature (float, optional): Defaults to 1.
        top_k (int, optional): Defaults to 0.
        top_p (float, optional): Defaults to 0.

    Returns:
        torch.Tensor: generated tokens (batch, target_seqlen)
    """
    seq_len = prefix.shape[1]
    T = seq_len + max_len

    assert prefix.shape[0] == 1, "input batch size must be 1"

    # with tqdm(total=T, desc="speculative sampling") as pbar:
    while prefix.shape[1] < T:
        # q = M_q[prefix + x_0, x_1, .., x_(gamma-2)]
        x = prefix
        prefix_len = prefix.shape[1]
        for _ in range(gamma):
            # p.logits shape (batch, seq, vocab)
            q = approx_model(x).logits
            next_tok = sample(norm_logits(q[:, -1, :], temperature, top_k, top_p))
            x = torch.cat((x, next_tok), dim=1)

        # normalize the logits
        for i in range(q.shape[1]):
            q[:, i, :] = norm_logits(q[:, i, :], temperature, top_k, top_p)
        # p  = M_p[prefix + x_0, x_0, .., x_(gamma-1)]
        p = target_model(x).logits
        for i in range(p.shape[1]):
            p[:, i, :] = norm_logits(p[:, i, :], temperature, top_k, top_p)

        # n the end position of the valid prefix
        # x = x_[:prefix_len-1] + x_0, ... x_(gamma-1)

        is_all_accept = True
        n = prefix_len - 1
        for i in range(gamma):
            r = torch.rand(1, device=p.device)
            j = x[:, prefix_len + i]

            if r < torch.min(
                torch.tensor([1], device=q.device),
                p[:, prefix_len + i - 1, j] / q[:, prefix_len + i - 1, j],
            ):
                # accept, and update n
                print(
                    f"\033[32m{decoder.decode(torch.tensor([j])[0], skip_special_tokens=True)}\033[0m",
                    end="",
                )
                n += 1
            else:
                # reject
                print(
                    f"\033[41m{decoder.decode(torch.tensor([j])[0], skip_special_tokens=True)}\033[0m",
                    end="",
                )
                t = sample(max_fn(p[:, n, :] - q[:, n, :]))
                print(
                    f"\033[34m{decoder.decode(t[0], skip_special_tokens=True)}\033[0m",
                    end="",
                )
                is_all_accept = False
                break

        prefix = x[:, : n + 1]

        if is_all_accept:
            t = sample(p[:, -1, :])
            print(
                f"{decoder.decode(t[0], skip_special_tokens=True)}",
                end="",
            )

        prefix = torch.cat((prefix, t), dim=1)
        # pbar.update(n - pbar.n)
    return prefix


if __name__ == "__main__":

    draft_model = AutoModelForCausalLM.from_pretrained(
        "facebook/opt-125m",
        torch_dtype=torch.float16,
        device_map="auto",
    )

    target_model = AutoModelForCausalLM.from_pretrained(
        "facebook/opt-1.3b",
        torch_dtype=torch.float16,
        device_map="auto",
    )

    prompt = "Why did the US join WW2?"

    torch_device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(
        "facebook/opt-125m", trust_remote_code=True
    )
    top_k = 20
    top_p = 0.9
    torch.manual_seed(1)
    num_tokens = 20
    lookahead = 4
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(torch_device)
    output = speculative_decoding(
        input_ids,
        draft_model,
        target_model,
        num_tokens,
        tokenizer,
        lookahead,
        top_k=top_k,
        top_p=top_p,
    )

    print(f"\n{tokenizer.decode(output[0], skip_special_tokens=True)}")

#!/usr/bin/python3
from sys import argv
from huggingface_hub import hf_hub_download
from transformers import OPTForCausalLM, GPT2Tokenizer
from joblib import Parallel, delayed

REPO_IDs = [
	"facebook/opt-125m",
	"facebook/opt-350m",
	"facebook/opt-1.3b",
	"facebook/opt-2.7b",
	"facebook/opt-6.7b",
	"facebook/opt-13b",
	"facebook/opt-30b"
]

def cache_model(REPO_ID):
    model = OPTForCausalLM.from_pretrained(REPO_ID, cache_dir=f"{argv[1]}/cache")
    tokenizer = GPT2Tokenizer.from_pretrained(REPO_ID, cache_dir=f"{argv[1]}/cache")
    return f"{REPO_ID} cached!\n"

if __name__ == "__main__":
    if (len(argv) != 2):
        print(f"USAGE {argv[0]} CACHE_DIR")
    else:
        results = Parallel(n_jobs=2)(delayed(cache_model)(REPO_ID) for REPO_ID in REPO_IDs)
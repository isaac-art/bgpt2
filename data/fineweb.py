import os
import argparse
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm
import re

def write_datafile(filename, toks):
    """ 
    Saves token data as a .bin file, for reading in C.
    - First comes a header with 256 int32s
    - The tokens follow, each as a uint16
    """
    assert len(toks) < 2**31, "token count too large" # ~2.1B tokens
    header = np.zeros(256, dtype=np.int32)
    header[0] = 20240520 # magic
    header[1] = 1 # version
    header[2] = len(toks) # number of tokens after the header
    if not isinstance(toks, np.ndarray) or not toks.dtype == np.uint16:
        maxtok = 2**16
        assert all(0 <= t < maxtok for t in toks), "token dictionary too large for uint16"
        toks_np = np.array(toks, dtype=np.uint16)
    else:
        toks_np = toks
    print(f"writing {len(toks):,} tokens to {filename}")
    with open(filename, "wb") as f:
        f.write(header.tobytes())
        f.write(toks_np.tobytes())

# ---------------- text reversal helpers ----------------

# Tokenise into: (1) "wordish" chunks (letters/digits/underscores plus internal ' or -),
# or (2) single punctuation/symbol chars. Whitespace is discarded.
_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+(?:[\'-][A-Za-z0-9_]+)*|[^\sA-Za-z0-9_]")

# Punctuation/symbols that should NOT have a space AFTER them when joining (opening brackets/quotes)
_NO_SPACE_AFTER = set("([{<«“‘\"'")
# Punctuation/symbols that should NOT have a space BEFORE them when joining (closing / end punctuation)
_NO_SPACE_BEFORE = set(")]}>»”’\"'.,;:!?")

def reverse_words_with_detached_punct(text: str) -> str:
    # Split into tokens where punctuation is separate
    toks = _TOKEN_RE.findall(text)
    if not toks:
        return text

    toks.reverse()

    # Re-join with sensible spacing:
    out = []
    for tok in toks:
        if not out:
            out.append(tok)
            continue

        prev = out[-1]

        # If current token is a "closing" or end punctuation, don't add a space before it
        if tok in _NO_SPACE_BEFORE:
            out.append(tok)
        # If previous token is an "opening" punctuation, don't add a space after it
        elif prev in _NO_SPACE_AFTER:
            out.append(tok)
        else:
            out.append(" " + tok)

    return "".join(out)

# ------------------------------------------

parser = argparse.ArgumentParser(description="FineWeb dataset preprocessing")
parser.add_argument("-v", "--version", type=str, default="10B", help="Which version of fineweb to use 10B|100B")
parser.add_argument("-s", "--shard_size", type=int, default=10**8, help="Size of each shard in tokens")
args = parser.parse_args()

assert args.version in ["10B", "100B"], "version must be one of 10B, 100B"
if args.version == "10B":
    local_dir = "fineweb10B"
    remote_name = "sample-10BT"
elif args.version == "100B":
    local_dir = "fineweb100B"
    remote_name = "sample-100BT"

DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

fw = load_dataset("HuggingFaceFW/fineweb", name=remote_name, split="train")

enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>'] # end of text token

def tokenize(doc):
    # reverse word order with punctuation detached BEFORE tokenising
    reversed_text = reverse_words_with_detached_punct(doc["text"])

    tokens = [eot]
    tokens.extend(enc.encode_ordinary(reversed_text))

    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    return tokens_np.astype(np.uint16)

nprocs = max(1, os.cpu_count() - 2)
with mp.Pool(nprocs) as pool:
    shard_index = 0
    all_tokens_np = np.empty((args.shard_size,), dtype=np.uint16)
    token_count = 0
    progress_bar = None
    for tokens in pool.imap(tokenize, fw, chunksize=16):
        if token_count + len(tokens) < args.shard_size:
            all_tokens_np[token_count:token_count+len(tokens)] = tokens
            token_count += len(tokens)
            if progress_bar is None:
                progress_bar = tqdm(total=args.shard_size, unit="tokens", desc=f"Shard {shard_index}")
            progress_bar.update(len(tokens))
        else:
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"fineweb_{split}_{shard_index:06d}.bin")
            remainder = args.shard_size - token_count
            progress_bar.update(remainder)
            all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
            write_datafile(filename, all_tokens_np)
            shard_index += 1
            progress_bar = None
            all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
            token_count = len(tokens)-remainder

    if token_count != 0:
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(DATA_CACHE_DIR, f"fineweb_{split}_{shard_index:06d}.bin")
        write_datafile(filename, all_tokens_np[:token_count])

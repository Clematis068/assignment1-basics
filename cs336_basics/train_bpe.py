import os
import heapq
from typing import BinaryIO
import regex as re
import collections
import multiprocessing as mp
import time
import pickle
from functools import reduce

PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

class ReverseLexOrderPair:
    """
    Encapsulates (bytes, bytes) so that in a min-heap, the "largest in normal lex order"
    is treated as the smallest. Ensures that tie frequencies pop in reverse lex order.
    """

    def __init__(self, pair: tuple[bytes, bytes]):
        self.pair = pair

    def __lt__(self, other: "ReverseLexOrderPair") -> bool:
        # Invert normal order: self < other if self is > other (so larger lex sorts first).
        return self.pair > other.pair

    def __eq__(self, other: "ReverseLexOrderPair") -> bool:
        return self.pair == other.pair

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def pre_tokenize_chunk(chunk: str, special_pattern : re.Pattern | None) -> dict[tuple[bytes], int]:
    # coarse grain (special_token) (use pre_tokenize to split boundary(parallel optim))
    # -> PAT -> ("word to tuple") -> stat freq
    freqs : dict[tuple[bytes], int] = {}
    sub_chunks = special_pattern.split(chunk) if special_pattern else [chunk] #要去掉special token然后分别分块

    for sub_chunk in sub_chunks:
        for match in PAT.finditer(sub_chunk):
            match_bytes = tuple(bytes([b]) for b in match.group().encode("utf-8"))
            freqs[match_bytes] = freqs.get(match_bytes, 0) + 1
    # 分块的match分别编码乘bytes，加到freqs内
    return freqs

def merge_freq_dicts(dict1: dict[tuple[bytes], int], dict2: dict[tuple[bytes], int]) -> dict[tuple[bytes], int]:
    res = dict1.copy()
    for k, v in dict2.items():
        res[k] = res.get(k, 0) + v

    return res

def pre_tokenize(input_path: str, special_tokens: list[str]) -> dict[tuple[bytes], int]:

    num_processes = mp.cpu_count()
    pool = mp.Pool(num_processes)

    chunk_freqs = []
    special_pattern = re.compile("|".join(re.escape(tk) for tk in special_tokens)) if special_tokens else None

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        for start, end in zip(boundaries[:-1], boundaries[1:]): # 50， 100， 150 取 (50, 100)，间隔一个
            f.seek(start)
            chunk_bytes = f.read(end - start)
            chunk_str = chunk_bytes.decode("utf-8") # 编码
            chunk_freqs.append(pool.apply_async(pre_tokenize_chunk, (chunk_str, special_pattern))) # 同步

    pool.close()
    pool.join() # 主进程阻塞，这块不太懂

    freqs_dicts = [res.get() for res in chunk_freqs]
    """
    [
    {'hello': 2, 'world': 1},   # 第1个区块(chunk)的词频
    {'hello': 1, 'test': 1},    # 第2个区块的词频
    {'world': 1, 'python': 3}   # 第3个区块的词频
    ...
    ]
    """
    combined_freqs = reduce(merge_freq_dicts, freqs_dicts, {}) # 把每个区块的东西合并[x, y] [c, d] (x + c) : (y + d)，迭代所有
    return combined_freqs
"""
    {
    'hello': 3,   # (来自 2 + 1)
    'world': 2,   # (来自 1 + 1)
    'test': 1,
    'python': 3
    }
"""
# word freqs -> pair freqs and which pairs in symbol 
def get_pair_freqs(freqs : dict[tuple[bytes], int]) -> tuple[dict[tuple[bytes, bytes], int], dict[tuple[bytes, bytes], set[tuple[bytes]]]]:
    pair_freqs = collections.defaultdict(int)
    pairs_to_keys = collections.defaultdict(set)

    for words, freq in freqs.items():
        for i in range(len(words) - 1):
            pair = (words[i], words[i + 1])
            pair_freqs[pair] += freq
            pairs_to_keys[pair].add(words)

    return pair_freqs, pairs_to_keys # 第二个是所有的字节字母

def build_new_words(old : tuple[bytes], pair: tuple[bytes, bytes]) -> tuple[bytes]:
    # return new words list like ("s", "b") to ("sb")
    new_words = []
    i = 0
    while i < len(old):
        if i < len(old) - 1 and old[i] == pair[0] and old[i + 1] == pair[1]:
            new_words.append(old[i] + old[i + 1])
            i += 2
        else:
            new_words.append(old[i])
            i += 1

    return tuple(new_words)
# 处理合并后的词表，词对表的频率信息,以及取什么词作为合并的对象
def merge(freqs: dict[tuple[bytes], int], 
          pair_freqs: dict[tuple[bytes, bytes], int],
          pair_to_keys: dict[tuple[bytes, bytes], set[tuple[bytes]]],
          pair: tuple[bytes, bytes] #这个从堆里边获得
) -> set[tuple[bytes, bytes]]:
    changed_pair = set()
    keys_to_modify = pair_to_keys[pair].copy()

    for old_key in keys_to_modify:
        old_freq = freqs.pop(old_key)
        new_key = build_new_words(old_key, pair)

        #  从pair_freq 去掉old_freq
        for i in range(len(old_key)- 1):
            l, r = old_key[i], old_key[i + 1]
            pair_freqs[l, r] -= old_freq
            changed_pair.add((l, r))
            if pair_freqs[l, r] <= 0:
                del pair_freqs[l, r]
            pair_to_keys[l, r].discard(old_key)
        # new_key放到ptk里边
        for i in range(len(new_key) - 1):
            l, r = new_key[i], new_key[i + 1]
            pair_freqs[l, r] += old_freq # 频率共用
            changed_pair.add((l, r))
            pair_to_keys[l, r].add(new_key)

        freqs[new_key] = freqs.get(new_key, 0) + old_freq

    pair_to_keys[pair] = set()

    return changed_pair

def write_merges(merges, outpath):
    """Pickle the merges list to a binary file."""
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, "wb") as f:
        pickle.dump(merges, f)
    print(f"Saved {len(merges)} merges to {outpath}")


def write_vocab(vocab, outpath):
    """Pickle the vocab dict to a binary file."""
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, "wb") as f:
        pickle.dump(vocab, f)
    print(f"Saved vocabulary with {len(vocab)} tokens to {outpath}")
    
def train_bpe(input_path: str, 
    vocab_size: int,
    special_tokens: list[str],
    merges_outpath: str = None,
    vocab_outpath: str = None,) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    train_start_time = time.time()
    init_tokens = [tok.encode("utf-8") for tok in special_tokens] + [bytes([i]) for i in range(256)]
    vocab = {i: token for i, token in enumerate(init_tokens)}
    merges = []

    print("start train")
    start_time = time.time()
    freqs = pre_tokenize(input_path, special_tokens)
    print(f"Pre-tokenize: finished in {time.time() - start_time:.2f}s")

    print("init pair_freq")
    start_time = time.time()
    pair_freqs, pair_to_keys = get_pair_freqs(freqs)

    pair_selected = []
    for p, f in pair_freqs.items():
        if f > 0:
            heapq.heappush(pair_selected, (-f, ReverseLexOrderPair(p), p))

    print(f"Initial pair frequencies: finished in {time.time() - start_time:.2f}s")

    n_init_tokens = len(init_tokens)
    n_merges = vocab_size - n_init_tokens

    print("Merge: start")
    start_time = time.time()

    for i in range(n_init_tokens, n_init_tokens + n_merges):
        if not pair_selected:
            break
            
        while pair_selected:
            neg_freq, _, top_pair = heapq.heappop(pair_selected) # 合法的最大频的pair
            freq = -neg_freq
            # 处理可能的新值和一定有的旧值
            if pair_freqs.get(top_pair, 0) == freq:
                pair = top_pair # 找到这个最好的如果词频内和找到的信息一致，不用更新
                break
            if top_pair in pair_freqs and pair_freqs[top_pair] > 0: # 如果不一致，说明数据过期了要更新
                heapq.heappush(pair_selected, (-pair_freqs[top_pair], ReverseLexOrderPair(top_pair), top_pair))

        else:
            break

        if pair_freqs.get(pair, 0) <= 0:
            break
        # 合并
        vocab[i] = pair[0] + pair[1]
        merges.append(pair)

        # 影响的东西增量更新
        changed_pairs = merge(freqs, pair_freqs, pair_to_keys, pair)
        for cp in changed_pairs: # 改变的东西一定是新值要更新
            if cp in pair_freqs and pair_freqs[cp] > 0:#更新后的东西放到堆内继续合并
                heapq.heappush(pair_selected, (-pair_freqs[cp], ReverseLexOrderPair(cp), cp))

        if ((i > n_init_tokens) and ((i - n_init_tokens + 1) % 100 == 0)) or (
            i == n_init_tokens + n_merges - 1
        ):
            print(
                f"{i - n_init_tokens + 1}/{n_merges} merges completed (merge runtime: {time.time() - start_time:.2f}s)"
            )

    print(f"Merges completed in {time.time() - start_time:.2f}s")
    print(f"Training completed in {time.time() - train_start_time:.2f}s")

    # Optionally save merges and vocab
    if merges_outpath:
        write_merges(merges, merges_outpath)
    if vocab_outpath:
        write_vocab(vocab, vocab_outpath)

    return vocab, merges
"""
if __name__ == "__main__":
    (vocab, merges) = train_bpe(
        input_path="./data/TinyStoriesV2-GPT4-valid.txt",
        vocab_size=10000,
        special_tokens=["<|endoftext|>"],
        merges_outpath="./out/ts-valid-merges-2.txt",
        vocab_outpath="./out/ts-valid-vocab-2.txt",
    )
"""

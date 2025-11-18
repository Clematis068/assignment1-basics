import os
from collections import Counter
special_token = ["<|endoftext|>"]

def build_vocab():
    vocab = []
    for i in range(256):
        vocab.append(bytes([i]).decode("latin-1"))

    for i in special_token:
        if i not in vocab:
            vocab.append(special_token)

def split_chunk(text: str):
    words = text.split()
    vocab_freq = Counter(words)
    return vocab_freq

def get_pair_stat(tokens_freq: dict):
    stats = Counter()
    for seq, freq in tokens_freq.items():
        for i in range(len(seq) - 1):
            pair = (seq[i], seq[i + 1])
            stats[pair] += freq
    # token序列的频率变成词对频率
    return stats

def merge_pair(tokens_freq):
    pair_stats = get_pair_stat(tokens_freq)
    if not pair_stats:
        return pair_stats, tokens_freq
    merged_pair_lookup = {}
    best_pair = max(pair_stats, key=pair_stats.get)
    p0, p1 = best_pair
    for token_seq, freq in tokens_freq.items():
        merged = []
        i = 0
        while i < len(token_seq):
            if i < len(token_seq) - 1 and token_seq[i] == p0 and token_seq[i + 1] == p1:
                merged.append(p0 + p1)
                # 相应的在tok to id id to token加入该新词,词对表删除原来的(p0, p1)信息 
                i += 2
            else:
                merged.append(token_seq[i])
                i += 1
    
        
def merge_pair(tokens_freq, vocab, tok2id, id2tok, merges):
    """
    tokens_freq: { token_seq(tuple): freq }
    vocab: list, 初始包含 <|endoftext|> 和 256 byte chars
    tok2id, id2tok: token 与 id 的双向映射
    merges: list, 记录 BPE 的 merge 顺序
    """

    # --- Step 1: 统计词对表 ---
    pair_stats = get_pair_stat(tokens_freq)
    if not pair_stats:
        return pair_stats, tokens_freq, vocab, tok2id, id2tok, merges

    best_pair = max(pair_stats, key=pair_stats.get)
    p0, p1 = best_pair

    new_tokens_freq = {}
    new_token = p0 + p1  

    for token_seq, freq in tokens_freq.items():
        merged = []
        i = 0
        while i < len(token_seq):
            if i < len(token_seq) - 1 and token_seq[i] == p0 and token_seq[i + 1] == p1:
                merged.append(new_token)
                i += 2
            else:
                merged.append(token_seq[i])
                i += 1
        # token序列
        merged = tuple(merged)
        if merged in new_tokens_freq:
            new_tokens_freq[merged] += freq
        else:
            new_tokens_freq[merged] = freq

    if new_token not in tok2id:
        new_id = len(vocab)
        vocab.append(new_token)
        tok2id[new_token] = new_id
        id2tok[new_id] = new_token

    if best_pair in pair_stats:
        del pair_stats[best_pair]

    merges.append(best_pair)
    # 相应的在tok to id id to token加入该新词,词对表删除原来的(p0, p1)信息 

    return pair_stats, new_tokens_freq, vocab, tok2id, id2tok, merges
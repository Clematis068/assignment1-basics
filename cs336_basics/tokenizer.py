from cs336_basics import train_bpe
from collections.abc import Iterable, Iterator
import re
import pickle
class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.inv_vocab = {v: k for k, v in vocab.items()}
        self.merges = merges
        self.merges_dict = {merge: i for i, merge in enumerate(merges)} # 按原来训练的顺序排序，题目给的要求

        self.pretokenize_pat = train_bpe.PAT

        if special_tokens:
            self.special_tokens = sorted(special_tokens, key=len, reverse=True)
            self.special_pat = "(" + "|".join(re.escape(k) for k in self.special_tokens) + ")"
            next_id = max(self.vocab.keys()) + 1
            for token in special_tokens:
                token_bytes = token.encode("utf-8")
                if token_bytes not in self.inv_vocab:
                    self.vocab[next_id] = token_bytes
                    self.inv_vocab[token_bytes] = next_id
                    next_id += 1
        else:
            self.special_tokens = None
            self.special_pat = None

        self.encode_cache = {}
        self.cache_hit = 0

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)

        with open(merges_filepath, "rb") as f:
            merges = pickle.loads(f)
        
        return cls(vocab, merges, special_tokens)
    
    def _pretokenize(self, text: str) -> list[str]:
        """Splits text into 'pretokens' and builds an initial byte representation for each."""
        pretokens: list[str] = []

        for match in self.pretokenize_pat.finditer(text):
            match_str = match.group() # 返回match的tuple
            pretokens.append(match_str)

        return pretokens

    def encode(self, text: str) -> list[int]:
        if not self.special_tokens:
            return self._encode_chunk(text)
        
        special_chunks = re.split(self.special_pat, text)

        ids = []
        for part in special_chunks:
            if part in self.special_tokens:
                ids.append(self.inv_vocab[part.encode("utf-8")] ) # special_token处理
            else:
                ids.extend(self._encode_chunk(part))

        return ids
    
    def _encode_chunk(self, text: str) -> list[int]:
        pretokens = self._pretokenize(text)
        pretoken_reprs: dict[str, list[bytes]] = {}

        ids = []

        for p in pretokens:
            if p in self.encode_cache:
                ids.extend(self.encode_cache[p])
                self.cache_hit += 1
            else:
                # Each character → single bytes: e.g. "abc" -> [b'a', b'b', b'c']
                if p not in pretoken_reprs: # 把匹配到的字符串拆成bytes放到matchbytes，放到rep方便循环转bytes
                    match_bytes = list(bytes([b]) for b in p.encode("UTF-8"))
                    pretoken_reprs[p] = match_bytes
                # 按照rep内的规则合并分词，从词表内找tkid翻到缓存
                merged = self._merge_subword(pretoken_reprs[p])
                token_ids = [self.inv_vocab[subword] for subword in merged]
                self.encode_cache[p] = token_ids
                ids.extend(token_ids)

        return ids
    
    def _merge_subword(self, rep: list[bytes]) -> list[bytes]:
        """
        Given a list of subword units (bytes), repeatedly merges adjacent pairs
        in ascending rank order until no more merges are found.
        """
        while True:
            best_rank = float("inf")
            best_idx = None

            # Scan adjacent pairs
            for i in range(len(rep) - 1):
                pair = (rep[i], rep[i + 1])
                rank = self.merges_dict.get(pair) # 按训练的顺序找词对对应的rk
                if rank is not None and rank < best_rank:
                    best_rank = rank
                    best_idx = i

            # If no merges found, we're done
            if best_idx is None:
                return rep

            # Merge the best pair
            merged = rep[best_idx] + rep[best_idx + 1]  # Concatenate bytes
            rep = rep[:best_idx] + [merged] + rep[best_idx + 2 :]


    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Yields token IDs lazily from an iterable of strings (e.g., a file handle)."""
        for text in iterable:
            yield from self.encode(text) # 不会直接返回一个大的列表而是要多少返回多少，跟IO有关

    def decode(self, ids: list[int]) -> str:
        """Decodes a sequence of token IDs into text."""# vocab -> 词表查 -> 变bytes -> decode
        text = b"".join(self.vocab[id] for id in ids)
        return text.decode("UTF-8", errors="replace")
    



        
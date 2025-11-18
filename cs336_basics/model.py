from einops import einsum, reduce, rearrange
import math
import torch
import torch.nn as nn 

class Linear(nn.Module): # no bias
    def __init__(self, in_features: int, out_features: int, device: torch.device | None =None, dtype: torch.dtype | None =None):
        super().__init__()
        mean = 0
        std = math.sqrt(2 / (in_features + out_features))
        l = -3 * std
        r = 3 * std
        W = torch.empty((out_features, in_features), device=device, dtype=dtype)
        torch.nn.init.trunc_normal_(W, mean, std, l, r)
        self.weight = nn.Parameter(W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(self.weight, x, "d_out d_in, ... d_in -> ... d_out")

class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device | None =None, dtype: torch.dtype | None =None, **kwargs):
        super().__init__()
        mean = 0
        std = 1
        l = -3 * std
        r = 3 * std
        if kwargs.get("embedding_std", None) is not None:
            std = kwargs.get("embedding_std")
        W = torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        nn.init.trunc_normal_(W, mean, std, l, r)
        self.weight = nn.Parameter(W)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]
    
class rmsnorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device: torch.device | None =None, dtype: torch.dtype | None =None, **kwargs):
        super().__init__()
        self.eps = eps # train and upd
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_type = x.dtype
        x = x.to(torch.float32)

        rms = torch.sqrt(reduce(x ** 2, "... d -> ... 1", "mean") + self.eps)
        res = x * self.weight / rms
        return res.to(in_type)

def silu_activation(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)

class SWiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device: torch.device | None =None, dtype: torch.dtype | None =None, **kwargs):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a1 = self.w1(x)
        silu = silu_activation(a1)
        return self.w2(silu * self.w3(x))
    
class SiLU(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()

        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a1 = self.w1(x)
        silu = silu_activation(a1)
        return self.w2(silu)
    
class rope(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None):
        super().__init__()
        self.d_k = d_k
        self.seq_len = max_seq_len
        positions = torch.arange(max_seq_len, device=device).unsqueeze(1) # 行变列 (seqlen, 1)
        assert positions.shape == (max_seq_len, 1)
        freqs = torch.arange(0, d_k, 2, device=device) / d_k
        inv_freq = 1.0 / (theta ** freqs) # (d/ 2, )
        assert inv_freq.shape == (d_k // 2,)
        # 我们要求得到一个矩阵而不是数，所以就是element wise
        angles = positions * inv_freq #l, d/ 2
        assert angles.shape == (max_seq_len, d_k // 2)
        self.register_buffer("sin", angles.sin(), persistent=False)
        self.register_buffer("cos", angles.cos(), persistent=False)
        

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos_pos = self.cos[token_positions]
        sin_pos = self.sin[token_positions]
        #assert cos_pos.shape[-1] == self.d_k // 2 # ..., seq_len, dk //2 匹配三维的情况
        # assert cos_pos.shape == (x.shape[0], x.shape[1], self.d_k // 2) 多头输入的话会变
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        #assert x_even.shape == (x.shape[0], self.seq_len, self.d_k // 2)
        x_rota_even = x_even * cos_pos + -x_odd * sin_pos
        x_rota_odd = x_odd * cos_pos + x_even * sin_pos
    
        x_rota = rearrange([x_rota_even, x_rota_odd], "two ... -> ... two") # batch， seq, d_k // 2, 2(stack的结果)
        #assert x_rota.shape == (x.shape[0], self.seq_len, self.d_k // 2, 2)
        x_out = rearrange(x_rota, "... d1 d2 -> ... (d1 d2)")

        return x_out
    
def softmax(x : torch.Tensor, dim: int) -> torch.Tensor:
    x_max = x.max(dim=dim, keepdim=True).values
    x_exp = torch.exp(x - x_max)
    return x_exp / x_exp.sum(dim = dim, keepdim=True)

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.tensor, mask: torch.Tensor) -> torch.Tensor:
    d_k = Q.shape[-1]
    attention_scores = einsum(Q, K, "... seq_q d, ... seq_k d -> ... seq_q seq_k")
    attention_scores = attention_scores / math.sqrt(d_k)
    attention_scores = torch.where(mask, attention_scores, float("-inf"))
    attention_w = softmax(attention_scores, dim=-1) # q关注哪些k合理
    output = einsum(attention_w, V, "... seq_q seq_k, ... seq_k d -> ... seq_q d")
    return output

class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, device = None, dtype = None, **kwargs):
        super().__init__()
        self.wqkv = Linear(d_model, 3 * d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % num_heads == 0
        self.d_head = d_model // num_heads

    def forward(self, 
                x: torch.Tensor,
                token_positions: torch.Tensor | None = None,
                Rope: rope | None = None
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        qkv = self.wqkv(x)

        q, k, v = qkv.split(self.d_model, dim = 2)

        q = rearrange(q, "b s (h d) -> b h s d", h = self.num_heads)
        k = rearrange(k, "b s (h d) -> b h s d", h = self.num_heads)
        v = rearrange(v, "b s (h d) -> b h s d", h = self.num_heads)

        if Rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len)
            q = Rope(q, token_positions)
            k = Rope(k, token_positions)

        # casual性质
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal = 0)
        y = scaled_dot_product_attention(q, k, v, mask)
        y = rearrange(y, "b h s d -> b s (h d)")
        return self.output_proj(y)
    
class Block(nn.Module):
    def __init__(self, 
                 d_model: int, 
                 num_heads: int,
                 d_ff: int,
                 Rope: rope | None = None,
                 device: torch.device | None = None,
                 dtype = None,
                 **kwargs, ):
        super().__init__()
        self.Rope = Rope
        self.ln1 = rmsnorm(d_model, device, dtype)
        self.ln2 = rmsnorm(d_model, device, dtype)
        self.attn = CausalMultiHeadSelfAttention(d_model, num_heads, device, dtype)
        ffn_type = kwargs.get("ffn_type", "SWiGLU")

        if ffn_type == "silu":
            self.ffn = SiLU(d_model, d_ff, device, dtype)
        else:
            self.ffn = SWiGLU(d_model, d_ff, device, dtype)
        
    def forward(self, x):
        x = x + self.attn(self.ln1(x), self.Rope)
        x = x + self.ffn(self.ln2(x))

        return x
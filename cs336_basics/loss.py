import torch

def CrossEntropyLossn(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    max_logits = logits.max(dim = -1, keepdim = True).values
    logits_shift = logits - max_logits
    sum_exp = torch.exp(logits_shift).sum(dim = -1, keepdim = True)
    log_sum_exp = torch.log(sum_exp)
    log_probs = logits_shift - log_sum_exp
    target_log_probs = log_probs.gather(dim = -1, index = targets.unsqueeze(-1)).squeeze(-1) # gather Index（索引）的维度数量必须和 Input（输入）一致
    return -target_log_probs.mean()

@torch.compile
def CrossEntropyLoss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    max_vals, max_idx = logits.max(dim = -1, keepdim = True)
    target_logits = logits.gather(dim = -1, index = targets.unsqueeze(-1))
    shift = logits - max_vals
    sum_exp = shift.exp().sum(dim = -1, keepdim = True)
    log_sum_exp = sum_exp.log()
    return -((target_logits - max_vals) - log_sum_exp).mean()

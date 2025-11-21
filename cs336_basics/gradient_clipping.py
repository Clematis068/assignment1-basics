from collections.abc import Iterable
import torch
@torch.compile
def gradient_clip(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float):
    grads = [p.grad for p in parameters if p.grad is not None]

    if not grads:
        return torch.tensor(0.0)
    
    stacked = torch.stack([torch.norm(g.detach(), p = 2) for g in grads])
    allnorm  = torch.norm(stacked, p = 2)

    if allnorm > max_l2_norm:
        scale = max_l2_norm / (allnorm + 1e-6)
        for grad in grads:
            grad.detach().mul_(scale)

    return allnorm

# compile的原因，torch的多个操作，以及循环都能优化减少gpu和cpu通#
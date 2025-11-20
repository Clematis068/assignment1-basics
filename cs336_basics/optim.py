import torch
import torch.nn as nn
from collections.abc import Callable
import math
from typing import Optional

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")

        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]  # Get state associated with p.
                t = state.get("t", 0)  # Get iteration number from the state, or initial value.
                grad = p.grad.data  # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad  # Update weight tensor in-place.
                state["t"] = t + 1  # Increment iteration number.
        return loss
'''
if __name__ == "__main__":
    lrs = [1, 1e1, 1e2, 1e3]

    for lr in lrs:
        print(f"========= LR: {lr} ==========")
        weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
        opt = SGD([weights], lr=lr)
        for t in range(10):
            opt.zero_grad()  # Reset the gradients for all learnable parameters.
            loss = (weights**2).mean()  # Compute a scalar loss value.
            print(loss.cpu().item())
            loss.backward()  # Run backward pass, which computes gradients.
            opt.step()  # Run optimizer step.
'''
# 为什么不用SGD，Adam能够根据历史梯度信息判断。出现我们希望对不常出现的特征更新得多一点，对常出现的更新少一点
# 自动调整学习率
# m, v:分别记录了方向变化信息以及震荡程度(专注程度不同)
# beta1, beta2分别控制了m,v参考过去梯度的程度

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.95), eps: float = 1e-8, weight_decay: float = 0.1, **kwargs):
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)
        
    @torch.no_grad()
    def step(self, closure = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            b1, b2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]

                m = state.get("m", torch.zeros_like(p.data))
                v = state.get("v", torch.zeros_like(p.data))
                t = state.get("t", 1) # init 1

                grad = p.grad

                state["m"] = b1 * m + (1 - b1) * grad
                state["v"] = b2 * v + (1 - b2) * grad.pow(2)

                cor = lr * (math.sqrt(1 - b2**t) / (1 - b1**t))

                p.data.addcdiv_(state["m"], torch.sqrt(state["v"]) + eps, value = -cor)

                if wd != 0:
                    p.data.add_(p.data, alpha = -lr * wd) # 不加下划线不会改p而是会生成一个新tensor

                state["t"] = t + 1
        return loss 
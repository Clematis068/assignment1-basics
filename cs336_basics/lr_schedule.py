import math

def lr_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    if it < warmup_iters:
        return (it / warmup_iters) * max_learning_rate
    
    if it <= cosine_cycle_iters:
        steps = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        cos = math.cos(steps * math.pi)
        return min_learning_rate + 1 / 2 * (1 + cos) * (max_learning_rate - min_learning_rate)
    
    return min_learning_rate
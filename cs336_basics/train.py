import math
import torch
import os
import numpy as np
from numpy.typing import NDArray
import typing
import yaml
import json
import wandb
import wandb.wandb_run
import argparse
import time
from cs336_basics.model import Transformer
from cs336_basics.optim import AdamW
from cs336_basics.loss import CrossEntropyLoss
from cs336_basics.gradient_clipping import gradient_clip
from cs336_basics.lr_schedule import lr_schedule
class Dataloader:
    def __init__(self, 
                 dataset: NDArray,
                 batch_size: int,
                 context_length: int,
                 device: str):
        self.dataset = dataset
        self.batch_size = batch_size
        self.context_length = context_length
        self.device = device

        self.max_start_idx = len(dataset) - context_length # 看的是x，y还要顺延一位
        if self.max_start_idx < 0:
            raise ValueError("error idx < 0")
        
    def to_device_(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.device == 'cuda:0':
            return (
                x.pin_memory().to(device = self.device, non_blocking = False),
                y.pin_memory().to(device = self.device, non_blocking = False)
            )
        
        return x.to(device = self.device), y.to(device = self.device)
    
    def get_batch(self, ) -> tuple[torch.Tensor, torch.Tensor]:
        start_idx = torch.randint(0, self.max_start_idx, size = (self.batch_size,))
        start_idx = start_idx.tolist()
        x_list = [torch.from_numpy((self.dataset[i : i + self.context_length]).astype(np.int64)) for i in start_idx]
        y_list = [torch.from_numpy((self.dataset[i + 1 : i + self.context_length + 1]).astype(np.int64)) for i in start_idx]
        
        x_batch = torch.stack(x_list)
        y_batch = torch.stack(y_list)   

        return self.to_device_(x_batch, y_batch)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        return self.get_batch()
            
def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]):
    checkpoint = {
        'model_state_dict' : model.state_dict(),
        'optimizer' : optimizer.state_dict(),
        'iteration' : iteration
    }

    torch.save(checkpoint, out)

def load_checkpoint(src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes], model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    iteration = checkpoint['iteration']
    return iteration

class Logger:
    """Logger that handles console, file, and wandb logging"""

    def __init__(self, log_file: str | None = None, wandb_run: wandb.wandb_run.Run | None = None, resume: bool = False):
        self.log_file = log_file
        self.wandb_run = wandb_run

        # Initialize log file
        if self.log_file:
            os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
            # Only clear the file if not resuming from checkpoint
            if not resume:
                with open(self.log_file, "w") as _:  # Clear the file
                    pass

    def log_info(self, message: str | dict, console=True):
        """Log a message to console and/or file"""
        if isinstance(message, dict):
            message = self.format_metrics(message)

        if console:
            print(message)

        if self.log_file:
            with open(self.log_file, "a") as f:
                f.write(message + "\n")

    def log_metrics(self, metrics: dict):
        """Log metrics to wandb"""
        if self.wandb_run:
            self.wandb_run.log(metrics)

    def format_metrics(self, metrics_dict: dict) -> str:
        """Format metrics dictionary into a readable string"""
        return " | ".join(f"{key}: {value}" for key, value in metrics_dict.items())


class Config(dict):
    """Config object that allows attribute-style access to dictionary keys"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self

        # Convert nested dictionaries to Config objects
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = Config(value)


def load_config_from_file(config_path: str) -> dict:
    """Load configuration from a file and return as a dictionary"""
    ext = os.path.splitext(config_path)[1].lower()
    with open(config_path) as f:
        if ext == ".json":
            return json.load(f)
        elif ext in [".yaml", ".yml"]:
            return yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config file format: {ext}")
        
def load_config(config_path: str | None = None, base_config: dict | None = None) -> Config:
    """Load configuration from a file and detect runtime device"""
    # Start with base_config if provided (e.g. when resuming), otherwise load default
    if base_config is None:
        default_config_path = os.path.join(os.path.dirname(__file__), "./configs/default.yml")
        config = load_config_from_file(default_config_path)
    else:
        config = base_config

    # If user specified a config, override defaults
    if config_path:
        user_config = load_config_from_file(config_path)

        # Deep update the config
        for section, section_config in user_config.items():
            if section in config:
                config[section].update(section_config)
            else:
                config[section] = section_config

    config["run"]["run_id"] = config["run"]["run_id"].replace("<timestamp>", f"{int(time.time())}")

    # Detect device and dtype at runtime
    device = "cpu"
    if torch.cuda.is_available():
        if config["training"].get("device", None) is not None:
            device = config["training"]["device"]
        else:
            device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"

    print(f"Using device: {device}")

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    config["device"] = device
    config["dtype"] = str(dtype)  # Convert dtype to string for JSON serialization

    # Convert nested dictionary to Config object
    return Config(config)

def deep_set(config_dict, key_path: str, value):
    keys = key_path.split('.')
    d = config_dict
    for k in keys[:-1]:
        if k not in d or not isinstance(d[k], dict):
            d[k] = {}
        d = d[k]
    d[keys[-1]] = value
# 递归设置attr

def train(config: Config | None = None):
    if config is None:
        config = load_config()

    resuming = config.training.get("resume", False)
    start_step = 1

    run_dir = os.path.join(config.run.out_dir, config.run.run_id)
    config_outfile = os.path.join(run_dir, "config.json")
    log_file = os.path.join(run_dir, "log.txt")
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    
    latest_symlink = os.path.join(config.run.out_dir, "latest")
    if os.path.islink(latest_symlink) or os.path.exists(latest_symlink):
        os.remove(latest_symlink)
    os.symlink(os.path.abspath(run_dir), latest_symlink, target_is_directory = True)

    wandb_run = (
        None
        if not config.run.wandb_project
        else wandb.init(project = config.run.wandb_project,
                        id = config.run.run_id,
                        resume = "must" if resuming else None,
                        name=config.run.run_id,
                        config=config,
                        dir=run_dir,
                        tags=config.run.wandb_tags,)
    )

    logger = Logger(log_file, wandb_run, resuming)

    if resuming:
        logger.log_info(f"Resuming training from existing config: {config_outfile}")
    else:
        with open(config_outfile, "w") as f:
            json.dump(config, f, indent=2, default=lambda o: list(o) if isinstance(o, tuple) else o.__dict__)
        logger.log_info(f"Saved config to: {config_outfile}")

    device = config.device
    dtype = getattr(torch, config.dtype.split('.')[-1])

    train_data = np.memmap(config.data.train_data_path, dtype = np.uint16, mode = "r")
    valid_data = np.memmap(config.data.valid_data_path, dtype = np.uint16, mode = "r")

    model = Transformer(**config.model, device = device, dtype = dtype)
    model.to(device)

    print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}") # 返回张量的参数数量

    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {"params": decay_params, **config.optimizer},
        {"params": nodecay_params, **config.optimizer, "weight_decay": 0.0},
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(f"Decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    print(f"Non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

    optimizer = AdamW(optim_groups, **config.optimizer)

    # load checkpoint
    if resuming:
        checkpoint_path = config.training.resume_checkpoint
        logger.log_info(f"Loading checkpoint from: {checkpoint_path}")
        start_step = load_checkpoint(checkpoint_path, model, optimizer) + 1
        logger.log_info(f"Resuming training from step {start_step}")

    max_steps = config.training.max_steps
    batch_size = config.training.batch_size
    max_l2_norm = config.training.max_l2_norm
    eval_interval = config.training.eval_interval
    checkpoint_interval = config.training.checkpoint_interval
    grad_accum_steps = config.training.grad_accum_steps

    lr_max = config.training.lr_max
    lr_min = config.training.lr_min
    warmup_ratio = config.training.warmup_ratio
    warmup_iters = config.training.warmup_iters
    cosine_cycle_iters = config.training.cosine_cycle_iters

    if warmup_iters is False or warmup_iters is None:
        warmup_iters = int(warmup_ratio * max_steps)

    if cosine_cycle_iters is False or cosine_cycle_iters is None:
        cosine_cycle_iters = max_steps

    if linear_cycle_iters is False or linear_cycle_iters is None:
        linear_cycle_iters = max_steps

    if phase_two_iters is False or phase_two_iters is None:
        phase_two_iters = max_steps

    def evaluate(step: int, is_last_step: bool):
        n_eval_steps = config.training.eval_steps

        if is_last_step:
            n_eval_steps = n_eval_steps * 3

        model.eval()
        val_loader = Dataloader(train_data, batch_size, config.model.context_length, device)
        with torch.no_grad():
            val_loss = 0.0
            for _ in range(n_eval_steps):
                x, y = val_loader.get_batch()
                with torch.autocast(device_type = device, dtype = dtype):
                    logits = model(x)
                loss = CrossEntropyLoss(logits, y)
                val_loss += loss.item()
            val_loss /= n_eval_steps
        progress_str = get_progress_str(step, max_steps)

        # WandB metrics
        metrics = {
            "eval/loss": val_loss,
            "eval/perplexity": get_perplexity(val_loss),
            "eval/peak_memory": get_peak_memory(device),
            "step": step,
        }

        # Console + local file metrics
        display_metrics = {
            "step": progress_str,
            "v_loss": f"{val_loss:.4f}",
            "v_ppl": f"{get_perplexity(val_loss):.2f}",
            "mem": f"{get_peak_memory(device):.1f}MB",
        }

        logger.log_info(display_metrics)
        logger.log_metrics(metrics)

        # Restore to training mode
        model.train()
    
    if config.training.eval_before_training and not resuming:
        evaluate(0)

    train_loader = Dataloader(train_data, batch_size, config.model.context_length, device)
    x, y = train_loader.get_batch()

    model.train()
    for step in range(start_step, max_steps + 1):
        t0 = time.time()
        is_last_step = step == max_steps
        loss_accum = 0.0

        for _ in range(grad_accum_steps):
            with torch.autocast(device_type = device, dtype = dtype):
                logits = model(x)

            loss = CrossEntropyLoss(logits, y) / grad_accum_steps
            x, y = Dataloader(train_data, batch_size, config.model.context_length, device).get_batch()

            loss_accum += loss.detach()
            loss.backward()

        norm = gradient_clip(model.parameters(), max_l2_norm)

        lr = lr_schedule(step, lr_max, lr_min, warmup_iters, cosine_cycle_iters)

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()
        t1 = time.time()
        dt = t1 - t0
        tokens_per_sec = config.model.context_length * batch_size * grad_accum_steps / dt
        train_loss = loss_accum.item()
        progress_str = get_progress_str(step, max_steps)

        # WandB metrics
        metrics = {
            "train/loss": train_loss,
            "train/perplexity": get_perplexity(train_loss),
            "train/lr": lr,
            "train/grad_norm": norm,
            "train/tokens_per_sec": tokens_per_sec,
            "train/peak_memory": get_peak_memory(device),
            "step": step,
        }

        # Console + local file metrics
        display_metrics = {
            "step": progress_str,
            "t_loss": f"{train_loss:.4f}",
            "t_ppl": f"{get_perplexity(train_loss):.2f}",
            "lr": f"{lr:.4e}",
            "grad_norm": f"{norm:.2f}",
            "mem": f"{get_peak_memory(device):.1f}MB",
            "tok/sec": f"{int(tokens_per_sec):,}",
            "dt": f"{dt * 1000:.2f}ms",
        }

        logger.log_info(display_metrics)
        logger.log_metrics(metrics)

        if step % eval_interval == 0 or is_last_step:
            evaluate(step, is_last_step)

        if step % checkpoint_interval == 0 or is_last_step:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{step}.pt")
            save_checkpoint(model, optimizer, step, checkpoint_path)

            # Create symlink pointing `latest` to checkpoint_path (remove `latest` if it exists)
            latest_symlink = os.path.join(checkpoint_dir, "latest.pt")
            if os.path.islink(latest_symlink) or os.path.exists(latest_symlink):
                os.remove(latest_symlink)
            os.symlink(os.path.abspath(checkpoint_path), latest_symlink)

            logger.log_info(f"Saved checkpoint to: {checkpoint_path}")

    wandb.finish()

def get_progress_str(step, max_steps):
    return f"step {step}/{max_steps} ({step / max_steps * 100:.2f}%)"

def get_peak_memory(device):
    """Get peak memory usage in MB on the current device"""
    if device != "cuda":
        return 0

    peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)
    torch.cuda.reset_peak_memory_stats()
    return peak_memory

def parse_value(value_str: str):
    """Convert argparse arg to list, int, float, bool, or leave as string."""
    if value_str.strip().startswith("[") and value_str.strip().endswith("]"):
        content = value_str.strip()[1:-1].strip()
        return [parse_value(v.strip()) for v in content.split(",")] if content else [] # 去除两端指定任意个字符

    try:
        return int(value_str)
    except ValueError:
        pass

    try:
        return float(value_str)
    except ValueError:
        pass

    lower = value_str.strip().lower()
    if lower in ("true", "false"):
        return lower == "true"

    return value_str

def get_perplexity(loss):
    """Calculate perplexity from cross-entropy loss"""
    return math.exp(min(loss, 20))  # Cap at 20 to avoid overflow

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a transformer model")
    parser.add_argument("--config", type=str, help="Path to config file (optional)")
    parser.add_argument("--resume-from", type=str, help="Path to run directory to resume from")
    parser.add_argument("--override-config", type=str, help="Path to config file with values to override when resuming")
    parser.add_argument(
        "--override-param",
        action="append",
        default=[],
        help="Override a config param, e.g. model.d_model=512 (can be repeated)",
    )
    args = parser.parse_args()

    if args.resume_from:
        # Load config from the previous run
        resume_config_path = os.path.join(args.resume_from, "config.json")
        base_config = load_config_from_file(resume_config_path)
        base_config["training"]["resume"] = True
        base_config["training"]["resume_checkpoint"] = os.path.join(args.resume_from, "checkpoints/latest.pt")

        # Use the override config if provided, or None otherwise
        config = load_config(args.override_config, base_config=base_config)
    else:
        config = load_config(args.config)  # Will use default if args.config is None

    for override_str in args.override_param:
        if "=" not in override_str:
            raise ValueError(f"Invalid override: {override_str}, must be like key=val")

        key, raw_value = override_str.split("=", 1)
        value = parse_value(raw_value)
        deep_set(config, key, value)

    train(config)

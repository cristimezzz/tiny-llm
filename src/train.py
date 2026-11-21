"""训练循环：next-token prediction。

用法：
    python src/train.py --config configs/s10m.yaml                       # 新训练
    python src/train.py --config configs/s10m.yaml --resume              # 从 ckpt_last.pt 续训
    python src/train.py --config configs/debug.yaml --max_iters 50 --name smoke_debug   # 冒烟
    python src/train.py --config configs/s100m.yaml --stop_iter 3000     # 跑到 3000 步存档退出
    python src/train.py --config configs/debug.yaml --overfit --name overfit_debug      # 过拟合测试

除 --config / --resume / --overfit 外，任何 --键 值 都会覆盖 yaml 里的同名项（见 utils.load_config）。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import GPT, MODEL_PRESETS, GPTConfig  # noqa: E402
from utils import CSVLogger, get_batch, get_lr, load_config, plot_loss, resolve_mix, set_seed  # noqa: E402

LOG_FIELDS = ["step", "tokens", "train_loss", "val_loss", "val_smollm", "val_tinystories",
              "lr", "grad_norm", "tokens_per_sec", "mem_gb", "elapsed_s"]


def eta_hours(it: int, stop_iter: int, tokens_per_iter: int, tokens_per_sec: float) -> float:
    """剩余训练时间（小时）。按 stop_iter 算而不是 max_iters：
    用 --stop_iter 分段训练或 --resume 续训时，ETA 才是"到这次要停的地方"还剩多久。"""
    if tokens_per_sec <= 0:
        return float("inf")
    return max(stop_iter - it, 0) * tokens_per_iter / tokens_per_sec / 3600


def build_model(cfg: dict) -> GPT:
    """model 小节：preset 给出层/维/头，其余键（block_size、dropout……）直接传给 GPTConfig。"""
    m = dict(cfg["model"])
    preset = MODEL_PRESETS[m.pop("preset")]
    return GPT(GPTConfig(**preset, **m))


def data_paths(cfg: dict, split: str) -> dict[str, str]:
    """来源名 -> .bin 路径，例如 smollm -> data/processed/smollm_train.bin。"""
    d = cfg["data"]["dir"]
    return {s: os.path.join(d, f"{s}_{split}.bin") for s in cfg["data"]["sources"]}


@torch.no_grad()
def estimate_loss(model: torch.nn.Module, cfg: dict, mix: dict[str, float],
                  device: str) -> dict[str, float]:
    """在训练集（按 mix 混合）和每个来源的验证集上各取 eval_iters 个 batch 求平均 loss。

    返回 {"train", "val_smollm", "val_tinystories", "val"}，val = 各来源 val 按 mix 加权。
    每次 eval 用同一个种子的 rng → 每次评估的是同一批数据，前后的 loss 可以直接比较。
    """
    tc = cfg["train"]
    B, T = tc["batch_size"], cfg["model"]["block_size"]
    model.eval()  # 关掉 dropout（本项目 dropout=0，但要写对）
    out: dict[str, float] = {}
    jobs = [("train", data_paths(cfg, "train"), mix)]
    jobs += [(f"val_{s}", data_paths(cfg, "val"), {s: 1.0}) for s in cfg["data"]["sources"]]
    for key, paths, weights in jobs:
        rng = np.random.default_rng(cfg["seed"] + 1)
        losses = torch.zeros(tc["eval_iters"])
        for k in range(tc["eval_iters"]):
            x, y = get_batch(paths, weights, B, T, device, rng)
            with autocast(cfg, device):
                _, loss = model(x, y)
            losses[k] = loss.item()
        out[key] = losses.mean().item()
    out["val"] = sum(mix[s] * out[f"val_{s}"] for s in cfg["data"]["sources"])
    model.train()  # eval 完切回训练模式
    return out


def autocast(cfg: dict, device: str) -> torch.autocast:
    """bf16 混合精度：矩阵乘法用 bf16 算（快、省显存），参数和优化器状态仍是 fp32。
    bf16 的指数位和 fp32 一样多（8 位），表示范围相同，梯度不会下溢成 0，
    所以不需要 GradScaler（fp16 指数位只有 5 位，才需要把 loss 放大防下溢）。"""
    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[cfg["dtype"]]
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    return torch.autocast(device_type=device_type, dtype=dtype, enabled=dtype != torch.float32)


# ---------------------------------------------------------------- checkpoint


def save_checkpoint(path: str, model: GPT, optimizer: torch.optim.Optimizer, it: int,
                    best_val: float, cfg: dict, extra: dict) -> None:
    """保存模型 + 优化器 + 步数 + 随机数状态，足够从这一步无缝续训。
    先写临时文件再 os.replace（原子操作）：写到一半断电也不会损坏已有的 checkpoint。"""
    ckpt = {
        "model": model.state_dict(),  # 传入的是未 compile 的原模型，键名没有 _orig_mod. 前缀
        "optimizer": optimizer.state_dict(),  # AdamW 的 m、v：丢了续训会像重新 warmup 一样抖
        "iter": it,
        "best_val": best_val,
        "config": cfg,
        "rng": {
            "python": random.getstate(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        **extra,  # meta、数据 rng 状态、累计训练时间
    }
    tmp = path + ".tmp"
    torch.save(ckpt, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str, model: GPT, optimizer: torch.optim.Optimizer) -> dict:
    """恢复模型、优化器和随机数状态，返回整个 checkpoint 字典（含 iter、best_val 等）。"""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    random.setstate(ckpt["rng"]["python"])
    torch.set_rng_state(ckpt["rng"]["torch"])
    if ckpt["rng"]["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(ckpt["rng"]["cuda"])
    return ckpt


# ---------------------------------------------------------------- 训练


def prepare_run_dir(cfg: dict, resume: bool) -> Path:
    """新训练：目录已存在就报错（不覆盖已有结果）；续训：目录必须存在。"""
    run_dir = Path(cfg["out_dir"]) / cfg["name"]
    if resume:
        if not (run_dir / "ckpt_last.pt").exists():
            raise FileNotFoundError(f"{run_dir}/ckpt_last.pt 不存在，无法续训")
    else:
        if run_dir.exists():
            raise FileExistsError(f"{run_dir} 已存在。换一个 --name，或加 --resume 续训")
        run_dir.mkdir(parents=True)
        with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
        shutil.copy(Path(cfg["data"]["dir"]) / "meta.json", run_dir / "meta.json")
    return run_dir


def train(cfg: dict, resume: bool = False, overfit: bool = False) -> dict[str, float]:
    """完整训练流程。返回最后一次评估的 loss。"""
    tc = cfg["train"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(cfg["seed"])
    # TF32：fp32 的矩阵乘法用 Tensor Core 加速（尾数精度降到 10 位，对训练几乎无影响）
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    run_dir = prepare_run_dir(cfg, resume)
    with open(Path(cfg["data"]["dir"]) / "meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    mix = resolve_mix(cfg, meta)
    train_paths = data_paths(cfg, "train")

    model = build_model(cfg).to(device)
    print(f"参数量：非 embedding {model.num_params(True) / 1e6:.2f}M，"
          f"总计 {model.num_params(False) / 1e6:.2f}M；设备 {device}；混合权重 {mix}")
    optimizer = model.configure_optimizers(tc["weight_decay"], tc["lr"], tuple(tc["betas"]))

    # 数据采样用独立的 numpy rng，它的状态存进 checkpoint，续训后采样序列和不中断时一致
    data_rng = np.random.default_rng(cfg["seed"])
    it, best_val, elapsed = 0, float("inf"), 0.0
    if resume:
        ckpt = load_checkpoint(str(run_dir / "ckpt_last.pt"), model, optimizer)
        it, best_val, elapsed = ckpt["iter"], ckpt["best_val"], ckpt["elapsed_s"]
        data_rng.bit_generator.state = ckpt["data_rng"]
        print(f"从 step {it} 续训（best_val={best_val:.4f}）")
    start_it = it

    raw_model = model  # 保存 checkpoint 用原模型；compile 后的模型 state_dict 键名带 _orig_mod.
    use_compile = cfg["compile"]
    if use_compile and device != "cuda":
        # CPU 上的 torch.compile 要现场调用 C++ 编译器（Mac / 没装编译器的电脑会报错），
        # 而且对 CPU 训练几乎没有加速，所以没有 GPU 时直接关掉
        print("没有 GPU：关闭 torch.compile（CPU 上需要 C++ 编译器，且加速不明显）")
        use_compile = False
    if use_compile:
        model = torch.compile(model)  # 第一次前向会编译 1–2 分钟

    B, T, accum = tc["batch_size"], cfg["model"]["block_size"], tc["grad_accum"]
    tokens_per_iter = B * accum * T
    stop_iter = min(tc.get("stop_iter") or tc["max_iters"], tc["max_iters"])
    logger = CSVLogger(str(run_dir / "log.csv"), LOG_FIELDS)
    ckpt_tokens = [int(float(n)) for n in tc["ckpt_tokens"]]  # yaml 里可能写成 1.0e+8
    saved_tok = {n for n in ckpt_tokens if n <= it * tokens_per_iter}
    print(f"有效 batch {B}×{accum}×{T} = {tokens_per_iter:,} token/步；"
          f"step {it} → {stop_iter}（max_iters {tc['max_iters']}）")

    fixed_batch = get_batch(train_paths, mix, B, T, device, data_rng) if overfit else None

    def ckpt_extra() -> dict:
        return {"meta": meta, "data_rng": data_rng.bit_generator.state, "elapsed_s": elapsed}

    def evaluate() -> dict[str, float]:
        """评估 → 写日志 → 存 ckpt_last（和 ckpt_best）。"""
        nonlocal best_val
        losses = estimate_loss(model, cfg, mix, device)
        mem = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0.0
        logger.log({"step": it, "tokens": it * tokens_per_iter, "val_loss": losses["val"],
                    "val_smollm": losses.get("val_smollm"),
                    "val_tinystories": losses.get("val_tinystories"),
                    "lr": get_lr(it, tc), "mem_gb": mem, "elapsed_s": elapsed})
        per_source = ", ".join(f"{s} {losses[f'val_{s}']:.4f}"
                               for s in cfg["data"]["sources"])
        print(f"[eval] step {it}: train {losses['train']:.4f} | val {losses['val']:.4f} "
              f"({per_source})")
        if losses["val"] < best_val:
            best_val = losses["val"]
            save_checkpoint(str(run_dir / "ckpt_best.pt"), raw_model, optimizer, it, best_val,
                            cfg, ckpt_extra())
        save_checkpoint(str(run_dir / "ckpt_last.pt"), raw_model, optimizer, it, best_val,
                        cfg, ckpt_extra())
        return losses

    losses: dict[str, float] = {}
    t_last, tok_since = time.time(), 0
    while it < stop_iter:
        # 续训时 start_it 那一步已经评估过（ckpt 就是在那时存的），不再重复
        if it % tc["eval_interval"] == 0 and not (resume and it == start_it):
            losses = evaluate()
            t_last, tok_since = time.time(), 0  # 计时不包含评估时间

        # 学习率按 step 计算，续训时 it 恢复后自然接上
        lr = get_lr(it, tc)
        for group in optimizer.param_groups:
            group["lr"] = lr

        # 梯度累积：把一个大 batch 拆成 accum 个 micro-batch 依次前向/反向。
        # backward 会把梯度累加到 .grad 上，loss 先除以 accum，累加结果才等于大 batch 的平均梯度。
        # loss 全程留在显存里累加，只在写日志时 .item() 同步一次；
        # 原来每个 micro-batch 都 .item()，会强制 GPU 与 CPU 同步，拖慢训练。
        loss_total = torch.zeros((), device=device)
        for _ in range(accum):
            x, y = fixed_batch if overfit else get_batch(train_paths, mix, B, T, device, data_rng)
            with autocast(cfg, device):
                _, loss = model(x, y)
            (loss / accum).backward()
            loss_total = loss_total + loss.detach().float()  # 记录未除以 accum 的值，日志里才是真实 loss

        # 梯度裁剪：总范数超过 grad_clip 时整体等比缩小，防止偶发的大梯度把参数"打飞"。
        # 返回值是裁剪前的范数，记进日志方便排查不稳定。
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"]).item()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)  # 置 None 比填 0 省一次内存写
        it += 1
        tok_since += tokens_per_iter

        # 到达指定 token 数时额外存档，供"相同 token 数对比"
        for n in ckpt_tokens:
            if n not in saved_tok and it * tokens_per_iter >= n:
                saved_tok.add(n)
                save_checkpoint(str(run_dir / f"ckpt_tok{n // 1_000_000}M.pt"), raw_model,
                                optimizer, it, best_val, cfg, ckpt_extra())

        if it % tc["log_interval"] == 0 or it == stop_iter:
            now = time.time()
            dt = now - t_last
            elapsed += dt
            tok_s = tok_since / dt
            mem = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0.0
            eta_h = eta_hours(it, stop_iter, tokens_per_iter, tok_s)
            train_loss = (loss_total / accum).item()
            logger.log({"step": it, "tokens": it * tokens_per_iter, "train_loss": train_loss,
                        "lr": lr, "grad_norm": grad_norm, "tokens_per_sec": tok_s,
                        "mem_gb": mem, "elapsed_s": elapsed})
            print(f"step {it:>6} | loss {train_loss:.4f} | lr {lr:.2e} | gnorm {grad_norm:.2f} "
                  f"| {tok_s / 1000:.1f}k tok/s | mem {mem:.2f}GB | ETA {eta_h:.2f}h")
            t_last, tok_since = now, 0

    # 收尾：最后评估一次（同时存 ckpt_last），画图
    losses = evaluate()
    plot_loss(str(run_dir / "log.csv"), str(run_dir / "loss_curve.png"))
    print(f"完成：step {it}，best val {best_val:.4f}，训练用时 {elapsed / 60:.1f} 分钟，"
          f"输出在 {run_dir}/")
    return losses


def main() -> None:
    parser = argparse.ArgumentParser(description="训练 GPT", allow_abbrev=False)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true", help="从 runs/<name>/ckpt_last.pt 续训")
    parser.add_argument("--overfit", action="store_true", help="反复训练同一个 batch，loss 应趋近 0")
    args, overrides = parser.parse_known_args()
    cfg = load_config(args.config, overrides)
    train(cfg, resume=args.resume, overfit=args.overfit)


if __name__ == "__main__":
    main()

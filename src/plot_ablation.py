"""组员 B 的消融对比图：学习率 ×0.5 / ×2、只用 TinyStories，以及本机对照。

用法：
    python src/plot_ablation.py \
        --runs runs/s10m_ctl runs/s10m_lr5e-4 runs/s10m_lr2e-3 runs/s10m_tts \
        --out docs/figures/

输出：
    ablation_val.png          加权 val loss 叠图（横轴已训练 token）
    ablation_tinystories.png  TinyStories 验证集 loss 叠图（含只用 TinyStories 的 run）
    ablation.md               最终结果表

为什么不直接用 plot_compare.py：它默认每个 run 都有 val_smollm 和 val_tinystories 两列，
而「只用 TinyStories」的 run 没有 val_smollm，会在取列时 KeyError。这里少一个来源就显示 —。
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无界面后端
import yaml  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plot_compare import load_logs, plot_overlay  # noqa: E402
from train import build_model  # noqa: E402


def final_stats(run_dir: str, rows: list[dict[str, float]]) -> dict:
    """一个 run 的最终结果。缺少某个来源的 val 列时记 None，不报错。"""
    run = Path(run_dir)
    cfg = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
    model = build_model(cfg)  # 只为数参数，不加载权重
    last = [r for r in rows if "val_loss" in r][-1]

    speeds = [r["tokens_per_sec"] for r in rows if "tokens_per_sec" in r]
    speeds = speeds[len(speeds) // 10:]  # 去掉前 10%（含 compile 和 warmup）
    ckpt = run / "ckpt_best.pt"
    return {
        "lr": cfg["train"]["lr"],
        "sources": "+".join(cfg["data"]["sources"]),
        "params": model.num_params(False),
        "tokens": last["tokens"],
        "val_loss": last["val_loss"],
        "val_smollm": last.get("val_smollm"),
        "val_tinystories": last.get("val_tinystories"),
        "ppl": math.exp(last["val_loss"]),
        "minutes": last["elapsed_s"] / 60,
        "tok_s": sum(speeds) / len(speeds) if speeds else float("nan"),
        "mem_gb": max((r.get("mem_gb", 0.0) for r in rows), default=0.0),
        "ckpt_mb": ckpt.stat().st_size / 1e6 if ckpt.exists() else float("nan"),
    }


def _fmt(value: float | None, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def results_markdown(final: dict[str, dict]) -> str:
    head = ("| run | lr | 数据 | 总参数 | 训练 token | val（加权） | val SmolLM | val TinyStories "
            "| 困惑度 | 用时 | 平均 tok/s | 显存峰值 | ckpt 文件 |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    lines = []
    for n, s in final.items():
        ckpt = "—" if math.isnan(s["ckpt_mb"]) else f"{s['ckpt_mb']:.0f} MB"
        lines.append(
            f"| {n} | {s['lr']:.1e} | {s['sources']} | {s['params'] / 1e6:.2f}M "
            f"| {s['tokens'] / 1e9:.2f}B | {s['val_loss']:.3f} "
            f"| {_fmt(s['val_smollm'])} | {_fmt(s['val_tinystories'])} | {s['ppl']:.1f} "
            f"| {s['minutes']:.1f} min | {s['tok_s'] / 1e3:.0f}k | {s['mem_gb']:.1f} GB "
            f"| {ckpt} |")
    note = ("\n\nval 取 log.csv 最后一次评估；困惑度 = exp(val 加权)；"
            "tok/s 去掉前 10% 的步；— 表示该 run 没有这个数据来源。\n")
    return head + "\n".join(lines) + note


def main() -> None:
    p = argparse.ArgumentParser(description="消融实验对比图")
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", default="docs/figures/")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    logs = load_logs(args.runs)
    plot_overlay(logs, "tokens", "val_loss", str(out / "ablation_val.png"))
    plot_overlay(logs, "tokens", "val_tinystories", str(out / "ablation_tinystories.png"))

    final = {Path(d).name: final_stats(d, logs[Path(d).name]) for d in args.runs}
    md = results_markdown(final)
    (out / "ablation.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"图和表已写入 {out}")


if __name__ == "__main__":
    main()

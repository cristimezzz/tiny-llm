"""阶段 4 的测试：学习率调度、配置覆盖、同名拒绝、断点续训一致性。

运行：pytest tests/test_train.py -v
"""

import copy
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from train import eta_hours, train  # noqa: E402
from utils import get_lr, load_config  # noqa: E402

LR_CFG = {"lr": 1e-3, "min_lr": 1e-4, "warmup_iters": 10, "max_iters": 100}


def test_lr_schedule() -> None:
    lrs = [get_lr(i, LR_CFG) for i in range(120)]
    assert lrs[0] == pytest.approx(1e-4)  # warmup 第 0 步 = lr / warmup
    assert lrs[9] == pytest.approx(1e-3)  # warmup 结束达到峰值
    assert all(a < b for a, b in zip(lrs[:9], lrs[1:10]))  # warmup 单调升
    assert all(a >= b for a, b in zip(lrs[10:100], lrs[11:101]))  # cosine 单调降
    assert lrs[55] == pytest.approx((1e-3 + 1e-4) / 2)  # 正中间 = 平均值
    assert lrs[100] == lrs[119] == pytest.approx(1e-4)  # 之后停在 min_lr


def test_eta_uses_stop_iter() -> None:
    """ETA 按 stop_iter 算：分段训练/续训时，剩余时间不该按 max_iters 估。"""
    # 1000 token/步、1000 token/s：还剩 40 步 = 40 秒
    assert eta_hours(10, 50, 1000, 1000) == pytest.approx(40 / 3600)
    assert eta_hours(50, 50, 1000, 1000) == 0
    assert eta_hours(60, 50, 1000, 1000) == 0  # 已超过 stop_iter 也不会变成负数
    assert eta_hours(0, 50, 1000, 1000) == pytest.approx(50 / 3600)


def test_load_config_overrides() -> None:
    cfg = load_config(str(ROOT / "configs/s10m.yaml"),
                      ["--max_iters", "50", "--name", "smoke", "--compile", "false",
                       "--lr", "3e-4", "--grad_ckpt", "true"])
    assert cfg["train"]["max_iters"] == 50 and isinstance(cfg["train"]["max_iters"], int)
    assert cfg["name"] == "smoke" and cfg["compile"] is False
    assert cfg["train"]["lr"] == pytest.approx(3e-4)
    assert cfg["model"]["grad_ckpt"] is True
    with pytest.raises(KeyError):
        load_config(str(ROOT / "configs/s10m.yaml"), ["--no_such_key", "1"])


def test_configs_consistent() -> None:
    """三个正式配置只允许 preset、batch、lr、max_iters 不同，有效 batch 相同。"""
    cfgs = {n: load_config(str(ROOT / f"configs/{n}.yaml"), []) for n in ("s10m", "s50m", "s100m")}
    eff = {n: c["train"]["batch_size"] * c["train"]["grad_accum"] for n, c in cfgs.items()}
    assert len(set(eff.values())) == 1
    strip = []
    for c in cfgs.values():
        c = copy.deepcopy(c)
        c.pop("name")
        c["model"].pop("preset")
        for k in ("batch_size", "grad_accum", "lr", "min_lr", "max_iters"):
            c["train"].pop(k)
        strip.append(c)
    assert strip[0] == strip[1] == strip[2]


def _tiny_setup(tmp_path: Path) -> dict:
    """临时目录里造一份随机小数据 + 一个很小的配置（CPU 上几秒跑完）。"""
    data = tmp_path / "data"
    data.mkdir()
    rng = np.random.default_rng(0)
    for s in ("smollm", "tinystories"):
        for split in ("train", "val"):
            rng.integers(0, 256, 5000, dtype=np.uint16).tofile(data / f"{s}_{split}.bin")
    (data / "meta.json").write_text(json.dumps({"mix_weights": {"smollm": 0.7, "tinystories": 0.3}}))
    return {
        "name": "tiny", "out_dir": str(tmp_path / "runs"),
        "model": {"preset": "debug", "vocab_size": 256, "block_size": 32, "dropout": 0.0,
                  "bias": True, "grad_ckpt": False},
        "data": {"dir": str(data), "sources": ["smollm", "tinystories"], "mix": "natural"},
        "train": {"batch_size": 4, "grad_accum": 2, "max_iters": 10, "stop_iter": None,
                  "lr": 1e-3, "min_lr": 1e-4, "warmup_iters": 3, "weight_decay": 0.1,
                  "betas": [0.9, 0.95], "grad_clip": 1.0, "eval_interval": 5, "eval_iters": 2,
                  "ckpt_tokens": [1000], "log_interval": 1},
        "seed": 1337, "dtype": "float32", "compile": False,
    }


@pytest.fixture
def cpu_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """强制走 CPU：结果确定、与显卡无关。"""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def test_refuse_existing_run(tmp_path: Path, cpu_only: None) -> None:
    cfg = _tiny_setup(tmp_path)
    cfg["train"]["max_iters"] = 2
    train(copy.deepcopy(cfg))
    with pytest.raises(FileExistsError):
        train(copy.deepcopy(cfg))


def test_resume_matches_uninterrupted(tmp_path: Path, cpu_only: None) -> None:
    """跑 10 步 vs 跑到第 5 步存档、再续训到 10 步：最终 loss 应一致。"""
    cfg = _tiny_setup(tmp_path)
    full = train(copy.deepcopy(cfg))

    cfg_b = copy.deepcopy(cfg)
    cfg_b["name"] = "tiny_resume"
    cfg_b["train"]["stop_iter"] = 5
    train(copy.deepcopy(cfg_b))
    cfg_b["train"]["stop_iter"] = None
    resumed = train(copy.deepcopy(cfg_b), resume=True)

    for k in ("train", "val"):
        assert resumed[k] == pytest.approx(full[k], abs=1e-5)
    run = Path(cfg["out_dir"]) / "tiny_resume"
    steps = [int(line.split(",")[0]) for line in (run / "log.csv").read_text().splitlines()[1:]]
    assert steps[-1] == 10 and (run / "loss_curve.png").exists()
    assert (run / "ckpt_tok0M.pt").exists()  # 1000 token 阈值
    assert not math.isnan(resumed["val"])


def test_single_source_run(tmp_path: Path, cpu_only: None) -> None:
    """只有一个来源时（消融实验「只用 TinyStories」），评估打印和日志不能崩。"""
    cfg = _tiny_setup(tmp_path)
    cfg["name"] = "tiny_tts"
    cfg["data"]["sources"] = ["tinystories"]
    cfg["train"]["max_iters"] = 2
    cfg["train"]["eval_interval"] = 1
    losses = train(copy.deepcopy(cfg))

    assert "val_tinystories" in losses and "val_smollm" not in losses
    assert math.isfinite(losses["val"]) and math.isfinite(losses["train"])

    run = Path(cfg["out_dir"]) / "tiny_tts"
    rows = (run / "log.csv").read_text(encoding="utf-8").splitlines()
    header = rows[0].split(",")
    assert "val_smollm" in header  # 列保留，但这一来源没有值
    idx = header.index("val_tinystories")
    assert any(line.split(",")[idx] for line in rows[1:])

# 项目文档索引

B03「从零训练迷你语言模型」的全部文档。我们在 SmolLM-Corpus 1B 子集 + TinyStories 混合语料上，从零训练了 **0.01B / 0.05B / 0.1B** 三个 Decoder-only Transformer 并做了对比。

## 按需求找文档

| 我想…… | 看这里 |
|---|---|
| 不训练，直接打开界面看效果 | 根目录 [README.md](../README.md) 的「快速运行 demo」 |
| 从头复现（环境 → 数据 → 训练 → 生成） | 根目录 [README.md](../README.md) 第 1–6 节 |
| 看实验结论和图表 | [report/experiments.md](report/experiments.md) |
| 准备答辩 | defense/ 下的问答、演示方案、速查表（见下表） |
| 弄懂某一步的原理和实现细节 | design/ 下对应阶段的文档（见下表） |
| 在这个项目上继续开发 | [DEVELOPING.md](DEVELOPING.md) |

## 目录结构

```
docs/
  README.md        本索引
  DEVELOPING.md    继续开发指南
  design/          阶段设计文档 00–08
  report/          交付文档：数据、模型、实验
  defense/         答辩材料：问答、演示、速查、PPT
  results/         生成样例、盲评数据（脚本生成）
  figures/         图和结果表（脚本生成）
```

**design/ 和 report/ 的区别**：design/ 是写代码之前的计划（函数签名、实现要点、验收标准），后面附了实现记录，讲的是"怎么做、为什么这么做"。report/ 是最终交付的结论，讲的是"做出来是什么样"。两边数字不一致时，**以 report/ 为准**。

### design/：阶段设计文档

每个文件都按同一结构写：目标 → 前置条件 → 要写的文件与函数签名 → 实现要点 → 验收标准 → 常见坑 → 答辩可能被问的问题。

| 文件 | 内容 |
|---|---|
| [00-roadmap.md](design/00-roadmap.md) | 总路线、里程碑、模型规模、资源估算、交付物对照表 |
| [01-environment.md](design/01-environment.md) | 阶段 1：Miniconda + TUNA 镜像环境、GPU 自检 |
| [02-data.md](design/02-data.md) | 阶段 2：两个语料的下载、清洗、验证集切分、BPE-8192、按来源混合采样 |
| [03-model.md](design/03-model.md) | 阶段 3：手写 GPT（注意力、MLP、权重共享） |
| [04-training.md](design/04-training.md) | 阶段 4：训练循环、日志、checkpoint、资源估算 |
| [05-sampling.md](design/05-sampling.md) | 阶段 5：temperature / top-k / top-p 采样、命令行生成 |
| [06-experiments.md](design/06-experiments.md) | 阶段 6：实验设计（规模、采样参数、分词方式） |
| [07-app.md](design/07-app.md) | 阶段 7：交互界面（Gradio 备用版 + React 网页版） |
| [08-report.md](design/08-report.md) | 阶段 8：README、报告素材、答辩准备 |

### report/：交付文档

| 文件 | 内容 |
|---|---|
| [data.md](report/data.md) | 数据集说明：来源、许可、下载文件与 sha256、清洗与丢弃统计、分词器、token 数 |
| [model.md](report/model.md) | 模型结构图（含 Block 内张量形状）、按组件的参数量统计、训练设置表 |
| [experiments.md](report/experiments.md) | 实验 A/B/C/D 的结果、图表、生成样例、盲评和结论 |

### defense/：答辩

| 文件 | 内容 |
|---|---|
| [qa.md](defense/qa.md) | 答辩问题清单与参考回答 |
| [demo.md](defense/demo.md) | 现场运行方案：启动步骤、演示脚本与预期输出、兜底方案 |
| [glossary.md](defense/glossary.md) | 概念速查（每条 2–3 句），答辩前通读 |
| `slides/` | 答辩 PPT 生成脚本 `build_slides.py`、模板和它的 conda 环境（生成的 pptx 不进 git） |

### results/ 和 figures/：脚本生成，不要手改

这些文件都由脚本生成，手改了下次重新生成时会被覆盖。要改内容，就改脚本再重新运行（完整命令见 [report/experiments.md](report/experiments.md) 文末）。

| 文件 | 生成命令 |
|---|---|
| `results/samples_{A,B,C,progress}.{jsonl,md}` | `python src/eval_prompts.py --grid scale/sampling/progress --out docs/results/samples_*.jsonl` |
| `results/blind_eval/sheet.md`、`key.json` | 同上，`--grid scale` 时顺带生成。**打分前不要看 key.json** |
| `results/blind_eval/R1–R5.md` | 五位打分人手填的打分表（这五份是手写的，重新生成时注意不要覆盖） |
| `results/blind_eval/summary.md` | `python src/score_blind.py --sheets docs/results/blind_eval/R*.md --flip_repeat R3` |
| `figures/*.png`、`figures/results.md` | `python src/plot_compare.py --runs runs/s10m runs/s50m runs/s100m --out docs/figures/` |
| `figures/ckpt_eval.md`、`figures/bpb.md` | `python src/evaluate.py loss ...` / `python src/evaluate.py bpb ...` |
| `figures/app_*.png` | 界面截图，手动截（方法见根 README 开头的注释） |

`figures/` 的路径被网页界面（`app/common.py` 的 `FIGURES`）和 PPT 脚本直接引用，**不要移动或改名**。

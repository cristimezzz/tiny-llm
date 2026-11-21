# tiny-llm：从零训练迷你语言模型（B03）

上海交通大学 Fall 2026 计算机科学导论小组项目。我们**手写** Decoder-only Transformer（`src/model.py`，不依赖 HuggingFace 模型类），在 SmolLM-Corpus 1B 子集 + TinyStories 混合语料（16.3 亿 token，自训练 BPE-8192 词表）上，用 next-token prediction 分别训练了 **0.01B / 0.05B / 0.1B** 三个规模的模型，并做了模型规模、采样参数、分词方式三组对比实验和一个网页交互界面（另有 Gradio 备用版）。

![界面截图](docs/figures/app_compare.png)

<!-- 截图说明：运行 python app/server.py --preload all，在“02 三模型对比”区块输入 Once upon a time、默认参数点生成，
     截整页保存为 docs/figures/app_compare.png；再在“01 单模型生成”区块截一张存为 docs/figures/app_single.png -->

同一个 prompt、同样的采样参数（T=0.8，top-k=50，seed=42）下三个模型的续写：

| prompt | s10m（0.01B） | s50m（0.05B） | s100m（0.1B） |
|---|---|---|---|
| Once upon a time | , there was a little boy named Timmy. Timmy loved to play with his toy car all day long. One day, Timmy's mommy gave him a big package of a new toy car… | , there was a little boy named Timmy. Timmy loved to play with his toy cars and trucks. One day, Timmy's mommy gave him a big box of blocks… | , there was a little boy named Timmy… One day, Timmy's mom told him it was time to take a bath. Timmy didn't want to stop playing, but he knew he had to listen to his mom. |
| The French Revolution began | in the aftermath of the Revolution, and the Roman Empire was the first settlement of the Roman Empire… | in the afternoon of 1864, and the French Revolution resulted in the establishment of its own government. Under Louis XIV's rule… | in the aftermath of the Revolutionary War, and on 17 September 1789, the French Revolution was declared an international war of war… |

三个模型都能写出通顺的儿童故事；说明文要到 0.1B 才开始"像样"，但事实仍然经常是编的。

---

## 快速运行 demo

不训练、只想打开界面看效果，三步：

```bash
# 1. 建环境（只需一次；镜像配置见第 1 节）
conda env create -f environment.yml        # 没有 NVIDIA 显卡（Mac 等）改用 environment-cpu.yml
conda activate tiny-llm

# 2. 下载训练好的模型（只需一次）：从第 4 节的交大云盘链接下载 tiny-llm-weights.tar，放到项目根目录后
tar xf tiny-llm-weights.tar -C runs/      # 得到 runs/{s10m,s50m,s100m}/ckpt_best.pt

# 3. 启动界面（在项目根目录运行）
python app/server.py --preload all
```

然后在浏览器打开 <http://127.0.0.1:7860>（WSL2 里启动的话，直接用 Windows 的浏览器打开这个地址即可）。

- 不需要 Node.js / pnpm：前端已经构建好放在 `app/web/dist/`，字体也打包在里面，**断网也能用**。
- 没有 GPU 也能跑（自动用 CPU，s100m 约 25 tok/s）。
- 想先确认一切正常：`python src/demo_check.py`（检查 GPU、三个 checkpoint、生成结果是否与记录一致）。
- 网页打不开时的备用界面：`python app/app.py --preload all`（Gradio 版，功能相同，没有流式输出）。
- 页面里有什么、怎么用，见第 6 节；答辩现场的演示流程见 `docs/defense/demo.md`。

## 1. 环境

开发机：RTX 5060 Ti 8GB（WSL2）；正式训练：云端 RTX 5090 32GB。两者都是 Blackwell（sm_120），**必须 PyTorch ≥ 2.7、CUDA ≥ 12.8**。

1. 安装 Miniconda（安装包可从 TUNA 镜像 `https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/` 下载）。
2. 把 conda 和 pip 指向清华 TUNA 镜像。`~/.condarc`：

   ```yaml
   channels:
     - conda-forge
   show_channel_urls: true
   channel_priority: strict
   default_channels:
     - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
   custom_channels:
     conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
   ```

   ```bash
   conda clean -i
   pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
   ```

3. 创建环境并自检：

   ```bash
   conda env create -f environment.yml     # 只用 conda-forge，PyTorch 2.14.1 + CUDA 12.9
   conda activate tiny-llm
   python src/check_env.py                 # 检查 GPU、bf16、SDPA、torch.compile
   ```

详见 [docs/design/01-environment.md](docs/design/01-environment.md)。原生 Windows 上把配置里的 `compile` 设成 `false`。

**没有 NVIDIA 显卡（MacBook、无独显电脑）**：第 3 步改用 `conda env create -f environment-cpu.yml`（CPU 版 PyTorch，环境名同样是 `tiny-llm`），其余命令不变。`check_env.py` 会自动按 CPU 模式检查，训练时自动关闭 `torch.compile`。CPU 适合跑界面、生成、测试和 debug 模型，正式规模的训练还是要用 GPU。Mac 需要 Apple 芯片和 macOS 12+。详见 [01-environment.md 第 6.1 节](docs/design/01-environment.md)。

## 2. 数据

```bash
# 全量：下载约 3.5GB parquet，清洗、切验证集、训练 BPE-8192、编码成 uint16 的 .bin
python src/prepare.py --datasets smollm,tinystories --tokenizer bpe --vocab_size 8192 --shards all
# 快速版：每个来源只下 1 个分片（约 0.57GB，3 分钟左右），够跑 debug 模型
python src/prepare.py --datasets smollm,tinystories --tokenizer bpe --vocab_size 8192 --shards 1
```

- 需要至少 12GB 磁盘空间。全量用时约 25 分钟（编码约 13 分钟，6 核 CPU；下载看网速）。
- 输出在 `data/processed/`：`{smollm,tinystories}_{train,val}.bin`、`meta.json`、`tokenizer.json`（后两个已经在 git 里）。
- 全量统计：训练集 16.3 亿 token（SmolLM 11.46 亿 + TinyStories 4.85 亿，自然比例约 70:30），验证集 1054 万 token。

来源、许可、清洗规则和丢弃统计见 [docs/report/data.md](docs/report/data.md)。

## 3. 训练

| 配置 | 结构（层/维/头） | 总参数 | 训练 token | 5090 用时 | 5090 显存峰值 | 本地 8GB 显卡的覆盖参数 |
|---|---|---|---|---|---|---|
| `configs/debug.yaml` | 4 / 256 / 4 | 5.39M | ~2000 万 | 本地 5060 Ti 约 1 分钟 | 1.6GB（本地） | 不需要 |
| `configs/s10m.yaml` | 6 / 384 / 6 | 13.99M | 3 亿 | 4.6 分钟 | 9.2GB | `--batch_size 64 --grad_accum 4` |
| `configs/s50m.yaml` | 10 / 640 / 10 | 54.81M | 10 亿 | 45 分钟 | 10.4GB | `--batch_size 32 --grad_accum 8` |
| `configs/s100m.yaml` | 14 / 768 / 12 | 105.92M | 15 亿 | 1 小时 56 分 | 16.5GB | `--batch_size 16 --grad_accum 16` |

```bash
python src/train.py --config configs/s10m.yaml                                 # 正式训练（5090）
python src/train.py --config configs/s10m.yaml --batch_size 64 --grad_accum 4  # 本地 8GB 显卡
python src/train.py --config configs/debug.yaml --name my_smoke --max_iters 50 # 冒烟测试
python src/train.py --config configs/debug.yaml --name my_overfit --overfit    # 过拟合测试：loss 应接近 0
python src/train.py --config configs/s100m.yaml --resume                       # 从 runs/s100m/ckpt_last.pt 续训
```

- 配置里的任何键都可以用 `--键 值` 覆盖（如 `--max_iters`、`--lr`、`--name`）。覆盖 batch 时保持 `batch_size × grad_accum = 256`，有效 batch 就和正式训练相同（131072 token/步）。
- 输出在 `runs/<name>/`：`config.yaml`（完整配置副本）、`log.csv`（step、train/val loss、学习率、tok/s、显存）、`loss_curve.png`、`ckpt_last.pt`、`ckpt_best.pt`、`ckpt_tok{100,300,1000}M.pt`。
- 本地 5060 Ti 上 s10m 约 18 分钟、s50m 约 3.6 小时、s100m 约 9 小时（冒烟实测吞吐估算）。

## 4. 下载训练好的模型（不想自己训练）

checkpoint 不进 git。我们把三个正式模型导出成**仅权重**版本（去掉 AdamW 状态，只能生成、不能续训），共 701MB：

- 交大云盘：<https://pan.sjtu.edu.cn/web/share/06331c94b063e87b7ef93d24f815a991> → 下载 `tiny-llm-weights.tar`
- 在项目根目录解压：`tar xf tiny-llm-weights.tar -C runs/`，得到 `runs/{s10m,s50m,s100m}/ckpt_best.pt`

自己训练完也可以这样导出：`python src/export.py --runs runs/s10m runs/s50m runs/s100m --out release/`。

## 5. 命令行生成

```bash
python src/sample.py --run runs/s100m --prompt "Once upon a time"
python src/sample.py --run runs/s100m --prompt "Photosynthesis is the process" \
    --max_new_tokens 200 --temperature 0.8 --top_k 50 --top_p 0.95 --num_samples 3 --seed 42
python src/sample.py --run runs/s100m --ckpt ckpt_tok300M.pt --prompt "Once upon a time"  # 训练中途的模型
```

`--temperature 0` 是 greedy；`--top_k 0` 表示不限制。

## 6. 交互界面

```bash
python app/server.py                 # 网页界面，浏览器打开 http://127.0.0.1:7860
python app/server.py --preload all   # 启动时就加载三个正式模型（答辩演示用），约 0.74GB 显存
python app/server.py --port 8000     # 换端口
python app/app.py --preload all      # 备用：Gradio 版界面，功能相同（没有流式输出）
```

架构：`app/server.py`（FastAPI 后端，生成复用 `src/sample.py`）+ `app/web/`（React + shadcn/ui 前端）。生成结果通过 SSE 流式返回，文字逐 token 出现；同一 seed 下与 `src/sample.py` 命令行的结果完全一致。

页面是一个长页，左侧目录可跳转到五个区块：

- **01 单模型生成**：选 checkpoint（包括训练中途的 `ckpt_tok*.pt`），用预设或滑块调生成长度、temperature、top-k、top-p、seed；Enter 生成，生成中可以停止。
- **02 三模型对比**：同一 prompt、同一组参数、同一 seed，0.01B / 0.05B / 0.1B 依次流式输出。
- **03 Loss 曲线**、**04 对比图**、**05 结果表**：各 run 的训练曲线、`docs/figures/` 里的实验图和结果表。

运行界面只需要 `tiny-llm` 环境：前端构建产物 `app/web/dist/` 已提交，`node_modules/` 不提交。修改前端需要 Node.js + pnpm（`conda env create -f app/web/environment.yml`），步骤见 `app/web/README.md`。

## 7. 结果摘要

| 模型 | 总参数 | 训练 token | val loss（加权） | SmolLM | TinyStories | 困惑度 | 5090 训练时长 |
|---|---|---|---|---|---|---|---|
| s10m | 13.99M | 0.30B | 3.098 | 3.629 | 1.845 | 22.2 | 4.6 min |
| s50m | 54.81M | 1.00B | 2.543 | 3.016 | 1.426 | 12.7 | 45.2 min |
| s100m | 105.92M | 1.50B | 2.409 | 2.865 | 1.330 | 11.1 | 115.9 min |

![val loss](docs/figures/val_overlay_tokens.png)

- **实验 A（模型规模）**：参数越多 val loss 越低，但收益递减（s10m→s50m 降 0.555，s50m→s100m 只降 0.134）。相同已训练 token 数下大模型也更低。五人盲评：故事在 0.05B 就到顶，说明文到 0.1B 还在进步。
- **实验 B（采样参数）**：greedy / 低温度会整句复读，T=1.5 会编出不存在的词；top-k=50 能在高温下兜底。推荐 T=0.7–0.8 + top-k=50。
- **实验 C（字符级 vs BPE）**：不同分词器的 loss 不能直接比，要换成 bits-per-byte。相同 token 预算下 BPE 更好（SmolLM 1.367 vs 1.602 bpb）。
- **实验 D（训练设置）**：s10m 上 lr 2e-3 比 1e-3 好（3.094 → 2.975），5e-4 欠拟合（3.256）；只用 TinyStories 把 TinyStories val 从 1.849 降到 1.559，但失去 SmolLM 能力。本机对照与云端 s10m 只差 0.13%。

完整结果、图表、样例和盲评见 [docs/report/experiments.md](docs/report/experiments.md)。模型结构和参数量见 [docs/report/model.md](docs/report/model.md)。

## 8. 测试

```bash
pytest tests/        # 数据、模型（形状/因果性/注意力等价/权重共享/过拟合）、训练、采样、评估、导出
python src/model.py  # 打印四档配置的参数量（按组件拆分）
```

## 目录结构

```
configs/      实验配置（debug / s10m / s50m / s100m / s10m_char）
src/          tokenizer.py  prepare.py  model.py  train.py  sample.py  utils.py
              evaluate.py  eval_prompts.py  plot_compare.py  score_blind.py  export.py  demo_check.py  check_env.py
app/          server.py（FastAPI 后端）  web/（React + shadcn/ui 前端）  app.py（Gradio 备用界面）  common.py
tests/        pytest 测试
data/         raw/（下载的 parquet，不进 git）  processed/（bin 不进 git；meta.json、tokenizer.json 进 git）
runs/         每次训练的输出（checkpoint 不进 git）
docs/         design/（阶段文档 00–08）  report/（数据、模型、实验）  defense/（问答、演示、PPT）
              results/（生成样例、盲评）  figures/（图和结果表）
```

文档索引见 [docs/README.md](docs/README.md)；要在这个项目上继续开发，先读 [docs/DEVELOPING.md](docs/DEVELOPING.md)。

## 数据许可

- SmolLM-Corpus 1B 子集（[ecreeth/1b-smollm-corpus](https://huggingface.co/datasets/ecreeth/1b-smollm-corpus)，源自 HuggingFaceTB/smollm-corpus）：**ODC-By 1.0**，使用时需署名。
- TinyStories（[roneneldan/TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)）：**CDLA-Sharing-1.0**，再分发数据需使用同样的许可。
- 本仓库不包含原始数据，只包含在它们上面训练出的分词器和统计信息。

## 参考文献

1. Vaswani et al. (2017). *Attention Is All You Need*. NeurIPS. https://arxiv.org/abs/1706.03762
2. Karpathy, A. *nanoGPT*. https://github.com/karpathy/nanoGPT
3. HuggingFace LLM Course, Chapter 7.6: Training a causal language model from scratch. https://huggingface.co/learn/llm-course/en/chapter7/6
4. https://dujh22.github.io/Dujinhua_wiki/aml2024/
5. Eldan, R. & Li, Y. (2023). *TinyStories: How Small Can Language Models Be and Still Speak Coherent English?* https://arxiv.org/abs/2305.07759
6. Ben Allal, L. et al. (2024). *SmolLM-Corpus*. https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus
7. Lozhkov, A. et al. (2024). *FineWeb-Edu*. https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
8. Hoffmann, J. et al. (2022). *Training Compute-Optimal Large Language Models*（Chinchilla）. https://arxiv.org/abs/2203.15556

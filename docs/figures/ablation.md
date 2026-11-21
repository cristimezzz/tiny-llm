| run | lr | 数据 | 总参数 | 训练 token | val（加权） | val SmolLM | val TinyStories | 困惑度 | 有效训练 | 墙钟 | 平均 tok/s | 显存峰值 | ckpt 文件 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s10m_ctl | 1.0e-03 | smollm+tinystories | 13.99M | 0.30B | 3.094 | 3.621 | 1.849 | 22.1 | 33.6 min | 33.3 min | 149k | 4.7 GB | 168 MB |
| s10m_lr5e-4 | 5.0e-04 | smollm+tinystories | 13.99M | 0.30B | 3.256 | 3.795 | 1.982 | 25.9 | 35.4 min | 35.4 min | 141k | 4.7 GB | 168 MB |
| s10m_lr2e-3 | 2.0e-03 | smollm+tinystories | 13.99M | 0.30B | 2.975 | 3.490 | 1.758 | 19.6 | 35.6 min | 35.7 min | 140k | 4.7 GB | 168 MB |
| s10m_tts | 1.0e-03 | tinystories | 13.99M | 0.30B | 1.559 | — | 1.559 | 4.8 | 33.0 min | 329.5 min | 152k | 4.7 GB | 168 MB |

val 取 log.csv 最后一次评估；困惑度 = exp(val 加权)；tok/s 去掉前 10% 的步；— 表示该 run 没有这个数据来源。
「有效训练」按 训练 token / 平均 tok/s 估算，排除编译、评估和电脑休眠；「墙钟」是 elapsed_s。
墙钟远大于有效训练的 run：s10m_tts（本机训练期间电脑休眠/挂起，实际训练时间仍以「有效训练」为准）。

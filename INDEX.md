# 学习卡索引

| 卡片 | 路径 | 层级 | 状态 | 用途 |
|---|---|---|---|---|
| Variational Autoencoder | [VAE.md](representations/latent/VAE.md) | L0–L2 | learning | 理解概率潜变量、重参数化及其具身智能用途 |
| Conditional Variational Autoencoder | [CVAE.md](representations/latent/CVAE.md) | L0–L2 | learning | 理解条件内多模态生成、条件先验与动作 latent |
| Evidence Lower Bound | [ELBO.md](representations/latent/ELBO.md) | L0–L2 | learning | 理解 VAE/变分模型训练目标的数学来源 |
| Vector-Quantized VAE | [VQ-VAE.md](representations/latent/VQ-VAE.md) | L0–L2 | learning | 理解离散 codebook、straight-through 与具身 tokenization |
| Joint-Embedding Predictive Architecture | [JEPA.md](representations/visual/JEPA.md) | L0–L2 | learning | 理解 latent prediction、EMA teacher 与语义表征学习 |
| Key-Value Cache | [KV-Cache.md](architectures/transformer/KV-Cache.md) | L0–L2 | learning | 理解自回归推理中的 K/V 复用、复杂度与显存权衡 |
| Diffusion Transformer | [DiT.md](architectures/transformer/DiT.md) | L0–L2 | learning | 理解 latent patches、adaLN-Zero 与 Transformer 去噪 |
| Flow Matching | [Flow-Matching.md](action-modeling/flow-matching/Flow-Matching.md) | L0–L2 | learning | 理解条件速度回归、概率路径与 ODE 动作生成 |
| Recurrent State-Space Model | [RSSM.md](architectures/world-model/RSSM.md) | L0–L2 | learning | 理解 belief state、prior/posterior 与 latent imagination |
| Low-Rank Adaptation | [LoRA.md](architectures/transformer/LoRA.md) | L0–L2 | learning | 理解低秩权重更新、参数效率、初始化与合并部署 |

## 待创建的原子卡

- AutoEncoder
- Probability Distribution
- KL Divergence
- Jensen's Inequality
- Reparameterization Trick
- Posterior Collapse

## 待创建的 VLA/WAM 卡

- Action Chunking
- Diffusion Policy
- Latent Dynamics Model
- Value Function

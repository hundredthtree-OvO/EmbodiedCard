# 学习卡索引

| 卡片 | 路径 | 层级 | 状态 | 用途 |
|---|---|---|---|---|
| Markov Decision Process | [MDP.md](foundations/reinforcement-learning/MDP.md) | L0–L2 | learning | 理解状态、转移、奖励、策略与长期序列决策 |
| Return 与 Discount Factor | [Return-and-Discount-Factor.md](foundations/reinforcement-learning/Return-and-Discount-Factor.md) | L0–L2 | learning | 理解单步奖励如何累积为长期回报，以及折扣因子的时间尺度含义 |
| Value Function | [Value-Function.md](foundations/reinforcement-learning/Value-Function.md) | L0–L2 | learning | 理解 Return 的条件期望，以及 V、Q、Advantage 的关系 |
| Bellman Equation | [Bellman-Equation.md](foundations/reinforcement-learning/Bellman-Equation.md) | L0–L2 | learning | 理解价值的一步递推、固定点与 TD target 的来源 |
| Variational Autoencoder | [VAE.md](representations/latent/VAE.md) | L0–L2 | learning | 理解概率潜变量、重参数化及其具身智能用途 |
| Conditional Variational Autoencoder | [CVAE.md](representations/latent/CVAE.md) | L0–L2 | learning | 理解条件内多模态生成、条件先验与动作 latent |
| Evidence Lower Bound | [ELBO.md](representations/latent/ELBO.md) | L0–L2 | learning | 理解 VAE/变分模型训练目标的数学来源 |
| Vector-Quantized VAE | [VQ-VAE.md](representations/latent/VQ-VAE.md) | L0–L2 | learning | 理解离散 codebook、straight-through 与具身 tokenization |
| Joint-Embedding Predictive Architecture | [JEPA.md](representations/visual/JEPA.md) | L0–L2 | learning | 理解 latent prediction、EMA teacher 与语义表征学习 |
| Querying Transformer | [Q-Former.md](architectures/vla/Q-Former.md) | L0–L2 | learning | 理解 learnable queries、视觉 token 压缩与视觉—语言桥接 |
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

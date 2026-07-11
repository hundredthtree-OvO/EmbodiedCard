# EmbodiedCard

面向具身智能学习的渐进式 Markdown 知识卡仓库，重点覆盖 VLA、WAM/世界模型、机器人学习、动作生成、规划控制，以及理解这些方向所需的数学与深度学习基础。

每个核心概念对应一张主卡，复杂推导和前置知识拆分为原子子卡。卡片默认包含：

- L0：一分钟理解；
- L1：动机、直觉、结构和具身智能位置；
- L2：数学、数值例子、训练/推理、PyTorch 与失败模式；
- L3：按需补充的论文推导、源码映射与研究问题。

## 当前卡片

- [VAE：变分自编码器](representations/latent/VAE.md)
- [ELBO：证据下界](representations/latent/ELBO.md)

完整索引见 [INDEX.md](INDEX.md)，学习依赖见 [KNOWLEDGE_GRAPH.md](KNOWLEDGE_GRAPH.md)。

## 配套 Skill

仓库同时包含 [build-embodied-learning-cards](skill/build-embodied-learning-cards/SKILL.md) Skill，可用于生成、深入、比较、验证和组织后续卡片。

完整目录位于：

```text
skill/build-embodied-learning-cards/
├── SKILL.md
├── agents/openai.yaml
├── assets/card-template.md
├── references/
│   ├── card-taxonomy.md
│   ├── quality-checklist.md
│   └── source-policy.md
└── scripts/validate_card.py
```

下载仓库后，可单独复制这个 Skill 目录复用。验证卡片：

```bash
python3 skill/build-embodied-learning-cards/scripts/validate_card.py \
  representations/latent
```

## Markdown 约定

- 行内数学：`$...$`；
- 块数学：独占行的 `$$ ... $$`；
- 结构图：GitHub Mermaid；
- 链接：标准 Markdown 相对链接；
- 代码：默认使用 PyTorch；
- 不使用 Obsidian 专属 Wiki Link。

在 VS Code 中请使用 Markdown Preview，并确认设置 `markdown.math.enabled` 已开启。

## 计划中的学习路线

1. 数学基础：概率分布 → KL Divergence → ELBO；
2. 潜变量模型：AutoEncoder → VAE → CVAE → VQ-VAE；
3. VLA 动作生成：Action Chunking → Diffusion → Flow Matching；
4. 世界模型：Latent Dynamics → RSSM → Imagination Rollout → Value Model；
5. 机器人基础：SE(3) → FK/IK → 控制空间 → 闭环执行。

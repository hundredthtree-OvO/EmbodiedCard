---
title: Bellman Equation
aliases: [贝尔曼方程, Bellman Expectation Equation, Bellman Optimality Equation]
category: reinforcement-learning-foundation
level: beginner
status: learning
prerequisites: [MDP, Return-and-Discount-Factor, Value-Function]
related: [Dynamic-Programming, Temporal-Difference, Q-Learning, Actor-Critic]
embodied_roles: [value-backup, critic-target, planning-recursion, long-horizon-credit]
created: 2026-07-19
updated: 2026-07-20
---

# Bellman Equation（贝尔曼方程）

> 目标：理解 Bellman Equation 如何把长期 Value 拆成“一步 reward + 下一状态 Value”，分清期望方程、最优方程、Bellman backup 与实际学习损失。

## L0：一分钟理解

### 一句话定义

Bellman Equation 是价值函数的一致性条件：**从现在开始的长期价值，等于下一步奖励与折扣后的未来价值之和的期望。**

### 它解决什么问题

直接等待完整轨迹才能知道 Return，学习很慢。Return 自身具有递推结构：

```math
G_t=R_{t+1}+\gamma G_{t+1}.
```

对两侧取条件期望，就能用一步交互和下一状态估计来更新当前价值，这成为动态规划、TD、Q-learning 和 actor-critic 的共同骨架。

### 在具身智能中有什么用

- 用短 rollout 加尾部 value 估计长 horizon 结果；
- 在世界模型中对 imagined states 做价值回传；
- 为连续控制 critic 构造 TD target；
- 解释为什么 terminal 与 time-limit truncation 必须采用不同 bootstrap 处理。

### 记住这三点

1. Bellman 方程是 value 的递归关系，不是某一种神经网络架构。
2. 固定策略用加权期望；寻找最优策略时才使用 $\max$。
3. 一次 sample backup 或 TD MSE 只是近似求解方程的方法，不等于方程本身。

## L1：直觉与结构

### 1. 背景：长期问题为什么能做一步分解

在导航中，“当前位置的好坏”看似取决于很远的未来。但未来可拆成两部分：马上获得的 reward，以及到达下一状态后尚未兑现的长期价值。只要状态满足 Markov 性，下一状态已包含继续预测所需的信息，不必反复携带全部过去。

### 2. 设计因果链

```mermaid
flowchart LR
    G["长轨迹 Return G_t"] --> D["拆出一步 R_{t+1}"]
    D --> N["剩余 Return G_{t+1}"]
    N --> E["对未来随机性取期望"]
    E --> B["R_{t+1} + γV(S_{t+1})"]
    B --> F["Bellman fixed point"]
    F --> A["DP / TD / Q-learning / Actor-Critic"]
```

### 3. Equation、Backup 与 Algorithm 的区别

| 层次 | 含义 | 例子 |
|---|---|---|
| Bellman equation | 真 value 应满足的等式 | $v_\pi=\mathcal{T}_\pi v_\pi$ |
| Bellman backup | 用右侧构造一个新估计 | $v_{k+1}\leftarrow\mathcal{T}_\pi v_k$ |
| 学习算法 | 用模型或样本执行 backup | value iteration、TD(0)、Q-learning |
| 训练损失 | 让网络预测靠近 target | squared TD error |

### 4. 期望方程与最优方程

- **Bellman expectation equation**：策略 $\pi$ 已固定，对策略选动作的分布取平均；
- **Bellman optimality equation**：允许每个状态选最佳动作，因此使用最大值；
- 把固定策略的期望误写成 $\max$，就把“评估当前策略”偷偷换成了“假设未来总能选最优动作”。

### 5. 数据流与张量形状

| 对象 | 表格 MDP | 深度 RL batch |
|---|---|---|
| 当前状态 | 索引 $s$ | `states: [B, D]` |
| reward | $r(s,a,s')$ | `rewards: [B]` |
| 下一状态价值 | $v(s')$ | `next_values: [B]` |
| 终止标记 | terminal state | `terminated: [B]` |
| TD target | 标量 | `targets: [B]` |

### 6. 在系统中的位置

Bellman Equation 连接了四层：MDP 给出转移和奖励；Return 给出递推结构；Value 对未来取条件期望；RL 算法用模型或交互样本逼近这个固定点。

## L2：数学与实现

### 1. Bellman expectation equation

状态价值形式：

```math
v_\pi(s)
=\sum_a\pi(a\mid s)
\sum_{s',r}p(s',r\mid s,a)
\left[r+\gamma v_\pi(s')\right].
```

动作价值形式：

```math
q_\pi(s,a)
=\sum_{s',r}p(s',r\mid s,a)
\left[r+\gamma\sum_{a'}\pi(a'\mid s')q_\pi(s',a')\right].
```

### 2. Bellman optimality equation

最优状态价值：

```math
v_*(s)
=\max_a\sum_{s',r}p(s',r\mid s,a)
\left[r+\gamma v_*(s')\right].
```

最优动作价值：

```math
q_*(s,a)
=\sum_{s',r}p(s',r\mid s,a)
\left[r+\gamma\max_{a'}q_*(s',a')\right].
```

### 3. 期望方程如何从 Value 定义得到

从定义开始：

```math
v_\pi(s)=\mathbb{E}_\pi[G_t\mid S_t=s].
```

代入 Return 递推式：

```math
v_\pi(s)
=\mathbb{E}_\pi[R_{t+1}+\gamma G_{t+1}\mid S_t=s].
```

根据条件期望的线性性，并注意在下一状态继续执行同一策略时：

```math
\mathbb{E}_\pi[G_{t+1}\mid S_{t+1}=s']=v_\pi(s'),
```

所以：

```math
v_\pi(s)
=\mathbb{E}_\pi[R_{t+1}+\gamma v_\pi(S_{t+1})\mid S_t=s].
```

最后显式展开策略概率和环境转移概率，就得到前面的求和形式。每一层求和都对应一类随机性：先由策略选动作，再由环境产生 reward 和下一状态。

### 4. 矩阵形式

对有限状态 MDP，在固定策略下令 $P_\pi$ 为状态转移矩阵，$r_\pi$ 为期望一步奖励向量：

```math
\mathbf{v}_\pi
=\mathbf{r}_\pi+\gamma P_\pi\mathbf{v}_\pi.
```

移项得：

```math
(I-\gamma P_\pi)\mathbf{v}_\pi=\mathbf{r}_\pi.
```

若逆存在，则：

```math
\mathbf{v}_\pi=(I-\gamma P_\pi)^{-1}\mathbf{r}_\pi.
```

小型已知模型可以直接解线性方程；大型或未知模型通常使用迭代与采样方法。

### 5. Bellman operator 与固定点

定义策略 Bellman operator：

```math
(\mathcal{T}_\pi v)(s)
=\mathbb{E}_\pi[R_{t+1}+\gamma v(S_{t+1})\mid S_t=s].
```

真实 value 是它的固定点：

```math
v_\pi=\mathcal{T}_\pi v_\pi.
```

在有限折扣 MDP 且 $0\le\gamma<1$ 时，$\mathcal{T}_\pi$ 关于最大范数是 $\gamma$-压缩映射：

```math
\lVert\mathcal{T}_\pi v-\mathcal{T}_\pi u\rVert_\infty
\le\gamma\lVert v-u\rVert_\infty.
```

直觉上，每做一次 backup，旧估计的最大误差最多保留 $\gamma$ 倍，因此反复迭代收敛到唯一固定点。$\gamma=1$ 时不能直接套用这个结论，需要额外的 episodic 与终止条件。

### 6. 最小数值例子

某非终止状态 $s$ 每一步得到 reward $1$；之后以 $0.5$ 概率留在 $s$，以 $0.5$ 概率进入价值为 $0$ 的 terminal state。令 $\gamma=0.9$。

Bellman 方程为：

```math
v(s)=1+0.9\left(0.5v(s)+0.5\times0\right).
```

整理：

```math
(1-0.45)v(s)=1,
```

```math
v(s)=\frac{1}{0.55}\approx1.8182.
```

若从 $v_0(s)=0$ 开始做 backup：

```math
v_1=1,
\quad v_2=1.45,
\quad v_3=1.6525,
```

它会逐步接近 $1.8182$，展示了固定点迭代的含义。

### 7. 模型已知时的最小实现

```python
import torch


def bellman_q_backup(
    transition: torch.Tensor,
    reward: torch.Tensor,
    values: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """
    transition[s, a, s_next] = p(s_next | s, a)
    reward[s, a, s_next] = E[R | s, a, s_next]
    values[s_next] = V(s_next)
    returns Q backup with shape [num_states, num_actions]
    """
    next_values = values.view(1, 1, -1)
    return (transition * (reward + gamma * next_values)).sum(dim=-1)


def value_iteration(
    transition: torch.Tensor,
    reward: torch.Tensor,
    gamma: float = 0.99,
    num_steps: int = 100,
) -> torch.Tensor:
    values = torch.zeros(transition.shape[0], device=transition.device)
    for _ in range(num_steps):
        q_backup = bellman_q_backup(transition, reward, values, gamma)
        values = q_backup.max(dim=-1).values
    return values
```

### 8. 模型未知时的 TD critic loss

```python
import torch
import torch.nn.functional as F


def td_value_loss(
    values: torch.Tensor,
    rewards: torch.Tensor,
    next_values: torch.Tensor,
    terminated: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    # 真正的环境终止后没有未来回报；time-limit truncation 不应在此置 1。
    bootstrap_mask = 1.0 - terminated.to(values.dtype)
    targets = rewards + gamma * bootstrap_mask * next_values.detach()
    return F.mse_loss(values, targets)
```

### 9. 公式—代码对应

| 数学对象 | 代码对象 | 说明 |
|---|---|---|
| $p(s'\mid s,a)$ | `transition` | 已知模型下对下一状态求期望 |
| $r(s,a,s')$ | `reward` | 一步 reward，而非完整 Return |
| $\sum_{s'}p(\cdot)[r+\gamma V(s')]$ | `bellman_q_backup` | 完整模型 backup |
| $\max_a$ | `.max(dim=-1)` | optimality backup；策略评估不能随意使用 |
| $r+\gamma(1-d)V(s')$ | `targets` | 一次采样得到的 TD target |
| Bellman residual 的样本近似 | `F.mse_loss(values, targets)` | 训练 surrogate，不是 Bellman 方程定义 |

代码中出现 MSE，是因为只有 transition sample 时无法精确计算条件期望。`targets` 是 Bellman 右侧的随机样本，反复回归让预测逼近其条件均值。`detach()` 防止网络通过同时移动 target 来取巧；实践中还常配合慢更新 target network。

### 10. 训练与推理

#### 训练

- model-based DP：已知 $P$ 与 $R$，直接对所有后继状态求和；
- model-free TD：从 replay buffer 采样 transition，用单样本近似期望；
- actor-critic：critic 逼近 Bellman fixed point，actor 根据 value/Q/advantage 更新。

#### 推理

- value-based 离散控制可选 $\arg\max_a Q(s,a)$；
- 连续控制通常由 actor 产生动作，Q 用于训练或候选排序；
- MPC / world model 可用 value 作为有限 rollout 末端的 bootstrap。

### 11. 失败模式与常见误解

#### 固定策略评估误用 max

$\sum_a\pi(a\mid s)$ 评价当前策略，$\max_a$ 假设选择最优动作，二者解决不同问题。

#### Terminal 与 truncation 使用同一 mask

真实 terminal 后未来价值为零；仅因时间上限截断时，环境过程可能仍继续，应保留 bootstrap。

#### 让梯度穿过 TD target

预测和目标由同一网络产生时，通常对目标 stop-gradient；否则优化问题与半梯度 TD 不同，可能不稳定。

#### 把一次 TD target 当成精确 Bellman 右侧

单个 transition 只是条件期望的一次样本，噪声可能很大。

#### 忽略 function approximation 的不稳定性

非线性网络、bootstrap 与 off-policy 数据同时出现时可能发散，即所谓 deadly triad。target network、replay、双 Q 或保守目标是常见缓解方式。

#### 在 $\gamma=1$ 时仍声称压缩收敛

标准 $\gamma$-contraction 证明要求 $\gamma<1$。无折扣 episodic 情形需依赖恰当终止等额外条件。

#### Reward 时序索引错一位

$R_{t+1}$ 是执行 $A_t$ 后得到的 reward。数据管线若把它与错误的 state/action 对齐，会系统性污染 target。

## 自测

### 基础题

1. Bellman expectation equation 与 optimality equation 的动作聚合方式有何不同？
2. 为什么 Bellman 方程可以只看一步 reward 与下一状态 value？
3. TD target 中为什么通常要对 `next_values` 停止梯度？

### 理解题

1. “Bellman equation、Bellman backup、TD loss”分别属于哪一层？
2. 为什么 $\gamma<1$ 有助于固定点迭代收敛？
3. 为什么 time-limit truncation 通常仍需 bootstrap？

### 迁移题

机器人 rollout 固定采样 100 步，数据集中只有 `done`，没有区分任务成功与时间上限。说明它如何影响 Bellman target，并提出数据字段上的修正。

<details>
<summary>参考答案</summary>

**基础题**

1. Expectation equation 按固定策略的动作概率做加权和，即使用 $\sum_a\pi(a\mid s)$；optimality equation 假设每个状态都选价值最大的动作，因此使用 $\max_a$。前者评价给定策略，后者描述最优控制。
2. Return 满足 $G_t=R_{t+1}+\gamma G_{t+1}$。在 Markov state 上取条件期望后，剩余 Return 的条件期望就是下一状态 value，因此长期问题可递归成一步 reward 与下一状态 value。
3. TD 训练通常采用 semi-gradient：更新当前预测时把 bootstrap target 暂时视为常数。停止 `next_values` 的梯度可以防止网络通过同时移动预测和目标来降低损失，并避免产生与标准 TD 更新不同的梯度项；实践中常再配合 target network。

**理解题**

1. Bellman equation 是真实 value 应满足的固定点等式；Bellman backup 是把等式右侧作用于当前估计以构造新估计；TD loss 则用采样 transition 得到带噪 target，并通过回归近似执行这种 backup。
2. 当 $\gamma<1$ 时，Bellman operator 会把任意两个 value 估计间的最大差异至多缩小为原来的 $\gamma$ 倍。反复作用后误差持续收缩，从而趋向唯一固定点。$\gamma=1$ 时不能直接使用这一压缩论证。
3. Time-limit truncation 只表示采样或训练窗口结束，并不表示任务动力学进入没有未来 reward 的终止状态。若将其 value 置零会截掉真实的后续回报，因此通常应从截断后的状态 bootstrap；只有任务语义上的 terminal 才关闭 bootstrap。

**迁移题**

若把所有 `done` 都当真实终止，时间上限处会错误地把下一状态 value 置零，导致靠近截断边界的价值低估。应分别记录 `terminated` 与 `truncated`，只用前者关闭任务语义上的 bootstrap；padding 再使用独立 valid mask。

</details>

## 学习导航

### 前置卡片

- [Markov Decision Process](MDP.md)
- [Return 与 Discount Factor](Return-and-Discount-Factor.md)
- [Value Function](Value-Function.md)

### 原子子卡

- Bellman Expectation Equation（本卡覆盖）
- Bellman Optimality Equation（本卡覆盖）
- Bellman Operator（本卡覆盖）
- Fixed Point 与 Contraction Mapping（待创建）

### 对比卡片

- Policy Evaluation vs Control（待创建）
- [Dynamic Programming vs Temporal-Difference](Temporal-Difference-Learning.md#7-与相近方法的区别)（见 TD 卡）
- [Monte Carlo vs Temporal-Difference](Temporal-Difference-Learning.md#7-与相近方法的区别)（见 TD 卡）

### 下一张推荐卡

- [Temporal-Difference Learning](Temporal-Difference-Learning.md)：把 Bellman 的一步采样目标变成在线学习算法。

## 参考资料

1. Sutton, R. S., & Barto, A. G. *Reinforcement Learning: An Introduction*, 2nd ed., Chapters 3–4. [作者官网](http://incompleteideas.net/book/the-book-2nd.html)
2. Bellman, R. (1957). *Dynamic Programming*. Princeton University Press.
3. Sutton, R. S., & Barto, A. G. *Reinforcement Learning: An Introduction*. [MIT Press](https://mitpress.mit.edu/9780262352703/reinforcement-learning/)

## L3：论文与源码深入（待补充）

- 推导 $\mathcal{T}_\pi$ 与 $\mathcal{T}_*$ 的压缩映射性质；
- 区分 mean-squared Bellman error、mean-squared projected Bellman error 与 sampled TD loss；
- 研究 double-sampling problem 与 residual-gradient 方法；
- 分析 function approximation、off-policy、bootstrap 组成的 deadly triad。

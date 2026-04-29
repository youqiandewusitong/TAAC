# TAAC Baseline 创新点分析与上分路线（结合“低秩 Attention + 双优化器”框架）

> 目标：基于你当前 `baseline` 代码，不做空泛概念，给出**能直接实验、能做消融、能解释收益来源**的创新方向。

---

## 0. 先给结论（你现在的 baseline 已经很强）

你的代码已经覆盖了很多高分关键点：

- **双优化器分离（Sparse=Adagrad, Dense=AdamW）**：已实现（`trainer.py`）
- **统一序列+特征 token 化并联合建模**：已实现（`model.py` 的 NS token + 多域序列 + HyFormer block）
- **可切换序列编码器（transformer/swiglu/longer）**：已实现
- **RankMixer 风格 NS tokenizer**：已实现
- **InfoNCE 混合损失**：已实现
- **高基数 embedding 周期重置**：已实现

所以你接下来不该做“从0到1大改”，而是做**1到1.5的结构创新 + 训练机制创新**，并通过消融拿分。

---

## 1. 当前 baseline 与你给的“TAAC硬核思路”对齐度

### 已对齐部分

1. **推荐≈LLM 的 token 统一建模**
   - 你已经把 user/item/int/dense/seq 都投影到 token 空间再统一处理。

2. **低秩/压缩思想**
   - `LongerEncoder` 做了 top-k 压缩；`RankMixer` 做了 token mixing。

3. **稀疏-稠密优化分治**
   - Sparse 用 Adagrad，Dense 用 AdamW，且支持 sparse reinit。

### 尚可创新的“缺口”

1. **缺少“优化器层面的自适应调度”**（目前是固定 lr）
2. **缺少“序列域之间的显式协同机制”**（现在主要在拼接后 mixer 协同）
3. **缺少“训练目标与线上AUC一致性的增强”**（主要仍是 pointwise BCE/Focal）
4. **缺少“难样本导向训练机制”**（hard negative / sample reweight）

---

## 2. 建议你重点做的 6 个创新点（按性价比排序）

## 创新点 A（优先级 S）：双优化器“异步调度”

### 思路
在已有 Adagrad+AdamW 基础上，不只是分开优化器，而是分开**学习率调度策略**：

- Dense(AdamW)：warmup + cosine decay
- Sparse(Adagrad)：前期高 lr 快速记忆，后期阶梯下降或按验证指标触发衰减

### 为什么可能提分
你的框架里 sparse 与 dense 收敛速度天然不同；固定 lr 往往让其中一侧过拟合/欠拟合。异步调度能让两侧更同步。

### 最小实验
- Baseline: `lr=1e-4, sparse_lr=0.05`
- Exp1: dense cosine, sparse constant
- Exp2: dense cosine, sparse step decay
- Exp3: dense cosine, sparse reduce-on-plateau(按val auc)

### 预期
AUC 更平滑，early stop 更晚触发但更高峰值。

---

## 创新点 B（优先级 S）：序列域可靠性门控（Domain Reliability Gating）

### 思路
4个域（a/b/c/d）质量不一定一致。给每个域输出一个可学习 gate（可依赖 `seq_len`, domain统计特征, 当前sample上下文），再融合到最终 `all_q`。

当前你是直接 `torch.cat(curr_qs)` 再线性投影；建议改为：

- 每域得到 `q_i`
- 学一个 gate `g_i in [0,1]`
- 融合 `q = concat(g_i * q_i)` 或 `q = sum(g_i * pooled(q_i))`

### 为什么可能提分
不同样本上，不同域信息密度不同（比如某域极短序列或噪声域）。门控可减少“坏域拖累”。

### 最小实验
- Exp1: 全局可学习标量 gate（每域一个）
- Exp2: sample-level gate（小MLP输入每域 pooled token + seq_len）
- Exp3: 加 entropy regularization 防止 gate塌缩

---

## 创新点 C（优先级 A）：Q-token 多样性正则（Anti-collapse Query Regularization）

### 思路
你每个域有 `num_queries` 个 query token，但可能塌缩到相似表示。加入轻量正则：

- 对同域 query token 做去相关（orthogonality / cosine diversity）
- 或对不同域 query 做互补约束

### 为什么可能提分
提升 query 子空间表达容量，尤其在 `num_queries > 1` 时能带来稳定收益。

### 最小实验
- only when `num_queries >=2`
- loss += `lambda_div * diversity_loss(q_tokens)`
- `lambda_div` 从 1e-4 ~ 1e-2 扫

---

## 创新点 D（优先级 A）：InfoNCE 从“全局单头”升级为“域感知对比学习”

### 思路
你现在有 hybrid loss（BCE/Focal + InfoNCE），但可以把正负样本构造做得更“推荐友好”：

1. 正样本：同 user 的短时正反馈 item
2. 难负样本：同类目高曝光未点击 item（in-batch + memory queue）
3. 域感知：对 domain a/b/c/d 各自产生对比子损失，再加权融合

### 为什么可能提分
纯 in-batch negative 往往过于随机，难样本密度不够。域感知对比会更贴近“语义补强”的目标。

### 最小实验
- Exp1: 仅 hard negative（不改结构）
- Exp2: 域感知 InfoNCE
- Exp3: temperature 分域可学习

---

## 创新点 E（优先级 A）：高基数 embedding 的“分层重置 + 冻结恢复”

### 思路
你已有每 epoch reinit 高基数 embedding。可升级为：

1. 按频次分桶：超低频/低频/中频/高频
2. 仅重置超低频桶
3. 重置后 K steps 冻结该桶 or 限幅更新

### 为什么可能提分
全量高基数重置可能破坏已学到的有效稀疏记忆；分层重置更精细。

### 最小实验
- Exp1: threshold 重置（现有）
- Exp2: 仅重置 tail 5% 频次 id
- Exp3: tail 重置 + 500 steps 冻结

---

## 创新点 F（优先级 B）：训练目标从纯 pointwise 向轻量 listwise 偏移

### 思路
不推翻当前 BCE/Focal，而是增加一个轻量 pairwise/listwise 项：

- 同一 user 的正负样本对做 BPR / pairwise logistic
- 与主损失线性组合

### 为什么可能提分
AUC是排序指标，纯 pointwise 对排序边界不够敏感；加一点 pairwise 常能“最后一公里”提分。

### 最小实验
- loss = BCE + λ * pairwise
- λ 从 0.05/0.1/0.2 扫

---

## 3. 你该怎么选：建议的“比赛节奏”

## 第一阶段（快测，2-3天）
只动训练策略，不动主干结构：

1. 创新点A（异步调度）
2. 创新点E（分层重置）
3. 创新点D-Exp1（hard negative）

目的：低风险拿到第一波稳定增益。

## 第二阶段（中改，3-5天）
小结构变动：

1. 创新点B（域门控）
2. 创新点C（query多样性正则）

目的：提升表示能力，争取显著提升。

## 第三阶段（冲榜）

1. 创新点D完整体（域感知InfoNCE）
2. 创新点F（轻量pairwise）
3. 最终融合/蒸馏（可选）

---

## 4. 消融实验矩阵（建议直接照抄）

| 实验ID | 改动 | 预期收益 | 风险 |
|---|---|---|---|
| E0 | baseline | 基线 | 无 |
| E1 | +A 异步调度 | 稳定收敛/AUC小幅提升 | 低 |
| E2 | +E 分层重置 | 降低稀疏震荡 | 低 |
| E3 | +D hard negative | 提升区分能力 | 中 |
| E4 | +B 域门控 | 中等提升 | 中 |
| E5 | +C query多样性 | 中等提升（num_queries>1） | 中 |
| E6 | +F pairwise项 | 排序指标提升 | 中高 |
| E7 | E1+E2+E4+E5 | 组合冲榜 | 中高 |

---

## 5. 你这套 baseline 最值得强调的“论文/答辩叙事”

你可以把创新叙事写成三层：

1. **统一建模层**：推荐问题 token 化，序列与特征交互同构进 attention 路径
2. **收敛控制层**：稀疏/稠密参数异步优化与分层重置，修复梯度统计失真
3. **语义增强层**：域感知门控 + 对比学习 + query多样性，提高表示密度

这比“我调了很多参数”更像高分方案。

---

## 6. 最后给你一个最实用的落地建议

如果你现在时间有限，优先做这三件：

1. **A 异步调度**（几乎零结构风险）
2. **B 域门控**（最符合你当前多域结构）
3. **D hard negative + InfoNCE增强**（最符合“补语义”）

这三项组合，通常能兼顾稳定性和上限，且和你当前代码天然兼容。

---

## 附：与当前代码的对应位置（便于你快速定位）

- 双优化器与训练流程：`baseline/trainer.py`
- 主模型与多域融合：`baseline/model.py`
- 超参与入口：`baseline/train.py`

（本文件按“先策略再实现”的方式写，后续如果你要，我可以再给你一版“逐文件修改清单 + patch级实现计划”。）

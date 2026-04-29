# 改进方案 #1: AMP混合精度训练 + 损失函数优化

**改进日期**: 2026-04-28  
**改进类型**: 训练优化 + 损失函数创新

---

## 一、改进内容概述

本次改进主要包含两个方面:

1. **AMP自动混合精度训练** - 减少显存占用,加速训练
2. **新增3种损失函数** - 针对类别不平衡和过拟合问题

---

## 二、技术细节

### 2.1 AMP混合精度训练

**原理**:
- 使用`torch.cuda.amp.autocast()`在前向传播时自动将部分操作转为FP16
- 使用`GradScaler`防止梯度下溢
- 保持关键操作(如损失计算)使用FP32精度

**优势**:
- **显存减少**: 约30-50%显存占用降低
- **速度提升**: 约1.5-2倍训练加速(在支持Tensor Core的GPU上)
- **精度保持**: 通过梯度缩放保持训练稳定性

**使用方法**:
```bash
# 默认开启AMP
python train.py --use_amp

# 关闭AMP(如遇到数值不稳定)
python train.py --no_amp
```

**代码位置**:
- [trainer.py:62-65](trainer.py#L62-L65) - 初始化GradScaler
- [trainer.py:407-442](trainer.py#L407-L442) - AMP训练循环

---

### 2.2 新增损失函数

#### 2.2.1 Dice Loss (推荐用于极度不平衡数据)

**原理**:
```
Dice = (2 * |X ∩ Y| + smooth) / (|X| + |Y| + smooth)
Loss = 1 - Dice
```

**优势**:
- 对类别不平衡非常鲁棒(比Focal Loss更强)
- 直接优化F1-score相关指标
- 适合正样本极少的场景(如CVR预测)

**使用方法**:
```bash
python train.py --loss_type dice --dice_smooth 1.0
```

**适用场景**: 正负样本比例 < 1:100

---

#### 2.2.2 Label Smoothing BCE (推荐用于防止过拟合)

**原理**:
```
y_smooth = y * (1 - ε) + 0.5 * ε
Loss = BCE(logits, y_smooth)
```

**优势**:
- 防止模型过度自信
- 提升泛化能力
- 对噪声标签有一定鲁棒性

**使用方法**:
```bash
python train.py --loss_type bce_smooth --label_smoothing 0.1
```

**推荐参数**: `label_smoothing=0.05~0.15`

---

#### 2.2.3 Focal Loss (已有,参数优化)

**原理**:
```
FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
```

**优势**:
- 自动降低易分类样本的权重
- 聚焦困难样本
- 适合中等不平衡场景

**使用方法**:
```bash
# 默认参数
python train.py --loss_type focal --focal_alpha 0.1 --focal_gamma 2.0

# 针对更不平衡的数据
python train.py --loss_type focal --focal_alpha 0.25 --focal_gamma 3.0
```

**参数调优建议**:
- `focal_alpha`: 正样本比例的倒数 (如正样本5%, 设为0.05)
- `focal_gamma`: 2.0(标准) → 3.0(更关注困难样本)

---

## 三、实验建议

### 3.1 快速测试方案

**Step 1**: 先用AMP + BCE测试基线性能
```bash
python train.py --use_amp --loss_type bce --batch_size 256
```

**Step 2**: 测试Focal Loss (适合中等不平衡)
```bash
python train.py --use_amp --loss_type focal --focal_alpha 0.1 --focal_gamma 2.0
```

**Step 3**: 测试Dice Loss (适合极度不平衡)
```bash
python train.py --use_amp --loss_type dice --dice_smooth 1.0
```

**Step 4**: 测试Label Smoothing (防止过拟合)
```bash
python train.py --use_amp --loss_type bce_smooth --label_smoothing 0.1
```

---

### 3.2 组合策略(高级)

可以尝试**损失函数加权组合**(需要修改代码):
```python
# 示例: Focal + Dice 组合
loss = 0.7 * focal_loss + 0.3 * dice_loss
```

---

## 四、预期效果

| 改进项 | 预期提升 | 备注 |
|--------|---------|------|
| AMP训练速度 | +50%~100% | 取决于GPU型号 |
| 显存占用 | -30%~50% | 可增大batch_size |
| Focal Loss (vs BCE) | AUC +0.5%~1.5% | 中等不平衡数据 |
| Dice Loss (vs BCE) | AUC +1%~3% | 极度不平衡数据 |
| Label Smoothing | 泛化能力提升 | 验证集更稳定 |

---

## 五、注意事项

1. **AMP兼容性**: 需要PyTorch >= 1.6, CUDA >= 10.0
2. **数值稳定性**: 如遇到NaN,尝试`--no_amp`或降低学习率
3. **损失函数选择**: 根据正负样本比例选择:
   - 1:10 → BCE或Focal
   - 1:50 → Focal (gamma=2~3)
   - 1:100+ → Dice
4. **超参数调优**: 建议先用默认参数,再根据验证集表现微调

---

## 六、下一步改进方向

- [ ] 对比学习辅助损失(利用序列特征)
- [ ] 多任务学习(同时预测点击+转化)
- [ ] 动态损失权重调整
- [ ] Poly Loss (最新改进版Focal Loss)

---

## 七、代码修改文件清单

- ✅ [trainer.py](trainer.py) - 添加AMP支持和新损失函数
- ✅ [utils.py](utils.py) - 实现Dice Loss和Label Smoothing BCE
- ✅ [train.py](train.py) - 添加命令行参数

---

**作者**: Claude (Amazon Q)  
**参考**: 腾讯广告算法大赛历年方案

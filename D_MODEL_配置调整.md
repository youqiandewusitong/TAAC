# 时间特征工程 - d_model 配置调整

## ⚠️ 重要变更

添加时间特征后，NS token 总数从 **65** 增加到 **80**（+15个时间特征）。

由于模型架构约束（RankMixer 要求 `d_model % T == 0`），需要调整 `d_model`。

## 🔧 已修改的配置

### 1. `baseline/train.py`
```python
# 原来
parser.add_argument('--d_model', type=int, default=64)

# 现在
parser.add_argument('--d_model', type=int, default=80)
```

### 2. `eval/infer.py`
```python
# 原来
_FALLBACK_MODEL_CFG = {
    'd_model': 64,
    ...
}

# 现在
_FALLBACK_MODEL_CFG = {
    'd_model': 80,
    ...
}
```

## 📊 为什么是 80？

### 计算公式
```
T = num_queries × num_sequences + num_ns
T = 1 × 4 + 76 = 80
```

其中：
- `num_queries = 1`
- `num_sequences = 4` (seq_a, seq_b, seq_c, seq_d)
- `num_ns = 76` (原来61 + 时间特征15)

### d_model 的有效值
对于 T=80，`d_model` 必须是 80 的因数：
- ✅ **80** (推荐)
- ✅ 160
- ❌ 64 (无法整除)

## 🚀 使用方法

### 训练（使用新的 d_model=80）
```bash
cd baseline
python train.py --data_dir ../data --d_model 80
```

### 推理（自动使用 d_model=80）
```bash
cd eval
python infer.py
```

## 📝 注意事项

1. **新训练的模型**: 使用 `d_model=80`，可以正常加载时间特征
2. **旧模型**: 如果有 `d_model=64` 的旧模型，需要重新训练
3. **参数量变化**: `d_model` 从 64 增加到 80，模型参数会略微增加（约 25%）

## 🔍 如果遇到错误

### 错误信息
```
d_model=64 must be divisible by T=80
```

### 解决方案
确保训练和推理都使用 `d_model=80`：
```bash
# 训练时明确指定
python train.py --d_model 80

# 或者使用默认值（已改为80）
python train.py
```

## ✅ 验证配置

运行以下命令验证配置正确：
```bash
cd baseline
python -c "
from train import parse_args
args = parse_args()
print(f'd_model = {args.d_model}')
print(f'Expected: 80')
"
```

---

**现在配置已经调整完毕，可以正常训练和推理了！** ✅

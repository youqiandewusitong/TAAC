# 时间特征工程优化说明

## 修改概述

已成功将时间特征工程集成到 `baseline/dataset.py` 中，确保不会出现维度错误和索引超出范围的问题。

## 新增时间特征

### 1. timestamp 特征（曝光时间）- 7个特征
- **year**: 年份（相对于2000年的偏移，范围：0-100）
- **month**: 月份（范围：0-12，0表示无效值）
- **day**: 日期（范围：0-31，0表示无效值）
- **hour**: 小时（范围：0-23）
- **minute**: 分钟（范围：0-59）
- **weekday**: 星期几（范围：0-6，0=周一，6=周日）
- **is_weekend**: 是否周末（范围：0-1，1表示周末）

### 2. label_time 特征（行为时间）- 7个特征
与 timestamp 特征相同的7个维度

### 3. 决策时间特征 - 1个特征
- **decision_time_bucket**: label_time - timestamp 的时间差分桶
  - 0: 无效或0秒
  - 1: 0-60秒
  - 2: 60-300秒（1-5分钟）
  - 3: 300-3600秒（5分钟-1小时）
  - 4: >3600秒（>1小时）

**总计：15个新增时间特征**（如果 label_time 列不存在，则只有7个特征）

## 特征安全性保证

### 1. 索引范围限制
所有时间特征都经过 `np.clip()` 处理，确保值在安全范围内：
```python
time_feats[:, 0] = np.clip(time_feats[:, 0], 0, 100)  # year: 0-100
time_feats[:, 1] = np.clip(time_feats[:, 1], 0, 12)   # month: 0-12
time_feats[:, 2] = np.clip(time_feats[:, 2], 0, 31)   # day: 0-31
time_feats[:, 3] = np.clip(time_feats[:, 3], 0, 23)   # hour: 0-23
time_feats[:, 4] = np.clip(time_feats[:, 4], 0, 59)   # minute: 0-59
time_feats[:, 5] = np.clip(time_feats[:, 5], 0, 6)    # weekday: 0-6
time_feats[:, 6] = np.clip(time_feats[:, 6], 0, 1)    # is_weekend: 0-1
```

### 2. 异常值处理
- 无效时间戳（<=0）自动填充为0（作为padding处理）
- 时间戳解析异常时使用默认值0
- 决策时间为负数时自动修正为0

### 3. Embedding词汇表大小
在 schema 中自动添加的时间特征词汇表大小：
```python
time_feature_vocab_sizes = [
    101,  # year (0-100)
    13,   # month (0-12)
    32,   # day (0-31)
    24,   # hour (0-23)
    60,   # minute (0-59)
    7,    # weekday (0-6)
    2,    # is_weekend (0-1)
    # 如果有 label_time，再加上：
    101, 13, 32, 24, 60, 7, 2,  # label_time 的7个特征
    5,   # decision_time_bucket (0-4)
]
```

## 代码修改详情

### 1. `dataset.py` 修改

#### 新增方法：`_extract_time_features()`
```python
def _extract_time_features(self, timestamps: np.ndarray) -> np.ndarray:
    """从Unix时间戳提取时间特征（UTC时区）
    
    返回形状为 (B, 7) 的数组，包含：
    [year, month, day, hour, minute, weekday, is_weekend]
    所有值都经过裁剪，确保在安全范围内，防止索引错误。
    """
```

#### 修改 `_convert_batch()` 方法
- 在处理 user_int_feats 之前提取时间特征
- 将时间特征追加到 user_int_feats 的末尾
- 自动检测 label_time 列是否存在

#### 修改 `_load_schema()` 方法
- 自动在 user_int_schema 中添加时间特征的定义
- 使用高 feature_id (10000+) 避免与现有特征冲突

#### 修改 buffer 预分配
- `_buf_user_int` 的维度增加了15（或7）以容纳时间特征

## 使用方法

### 训练时无需修改代码
时间特征会自动提取并添加到 `user_int_feats` 中，模型会自动对这些特征进行 embedding。

### 验证时间特征提取
运行测试脚本：
```bash
cd baseline
python test_time_features.py
```

## 注意事项

1. **时区**: 所有时间特征使用 UTC 时区，避免不同时区的差异
2. **特征顺序**: 时间特征始终追加在 user_int_feats 的末尾
3. **向后兼容**: 如果数据中没有 label_time 列，只会提取 timestamp 的7个特征
4. **性能**: 时间特征提取使用 numpy 向量化操作，性能开销很小

## 测试结果

所有测试通过：
- ✓ 时间特征值在安全范围内
- ✓ 不会出现 embedding 索引错误
- ✓ 决策时间分桶正确
- ✓ 异常值处理正确

## 模型训练建议

由于新增了15个时间特征，建议：
1. 检查模型的 embedding 参数量是否显著增加
2. 可以考虑对时间特征使用较小的 embedding 维度
3. 监控训练过程中是否有 OOB (out-of-bound) 警告

## 故障排查

如果遇到维度错误：
1. 检查 schema.json 是否正确加载
2. 确认 parquet 文件中包含 timestamp 列
3. 查看日志中的 buffer size 警告信息

如果遇到索引超出范围错误：
1. 检查 `_extract_time_features()` 中的 clip 操作是否正确
2. 确认 vocab_sizes 设置是否足够大
3. 运行 `test_time_features.py` 验证特征提取逻辑

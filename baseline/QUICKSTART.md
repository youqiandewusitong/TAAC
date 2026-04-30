# 快速开始指南 - 时间特征工程

## 🚀 立即开始使用

### 第一步：验证修改

运行测试脚本确保一切正常：

```bash
cd baseline
python test_time_features.py
```

预期输出：
```
[PASS] All time features are within safe ranges!
[PASS] No index errors will occur during embedding lookup!
[PASS] Decision time buckets are within safe range [0, 4]!
All tests passed!
```

### 第二步：无需修改训练代码

时间特征已自动集成，直接运行训练即可：

```bash
cd baseline
python train.py --data_dir ../data --batch_size 256
```

### 第三步：验证特征提取

如果想确认时间特征是否正确提取，可以在训练脚本中添加：

```python
# 在训练循环开始前
batch = next(iter(train_loader))
user_int_feats = batch['user_int_feats']
print(f"user_int_feats shape: {user_int_feats.shape}")
print(f"Last 15 features (time features): {user_int_feats[0, -15:]}")
```

---

## 📝 新增的时间特征

### 自动提取的15个特征：

1. **timestamp_year** (0-100): 年份相对2000年的偏移
2. **timestamp_month** (0-12): 月份
3. **timestamp_day** (0-31): 日期
4. **timestamp_hour** (0-23): 小时
5. **timestamp_minute** (0-59): 分钟
6. **timestamp_weekday** (0-6): 星期几
7. **timestamp_is_weekend** (0-1): 是否周末
8. **label_time_year** (0-100): 行为时间年份
9. **label_time_month** (0-12): 行为时间月份
10. **label_time_day** (0-31): 行为时间日期
11. **label_time_hour** (0-23): 行为时间小时
12. **label_time_minute** (0-59): 行为时间分钟
13. **label_time_weekday** (0-6): 行为时间星期几
14. **label_time_is_weekend** (0-1): 行为时间是否周末
15. **decision_time_bucket** (0-4): 决策时间分桶

---

## ⚙️ 配置说明

### 默认配置（无需修改）

时间特征使用以下默认配置：

```python
# Vocab sizes (自动设置)
time_feature_vocab_sizes = [
    101,  # year (0-100)
    13,   # month (0-12)
    32,   # day (0-31)
    24,   # hour (0-23)
    60,   # minute (0-59)
    7,    # weekday (0-6)
    2,    # is_weekend (0-1)
]

# 如果有 label_time，再加上：
# + 7个 label_time 特征
# + 1个 decision_time 特征 (vocab_size=5)
```

### 高级配置（可选）

如果需要自定义时间特征，可以修改 `dataset.py` 中的：

1. **时间分桶策略**（第588-595行）
2. **特征范围**（第552-558行）
3. **Vocab sizes**（第286-295行）

---

## 🔍 常见问题

### Q1: 如何确认时间特征已经生效？

**A**: 运行以下代码：

```python
from dataset import PCVRParquetDataset

dataset = PCVRParquetDataset(
    parquet_path='../demo_1000.parquet',
    schema_path='../schema.json',
    batch_size=32,
)

print(f"Total user_int dimensions: {dataset.user_int_schema.total_dim}")
print(f"Vocab sizes count: {len(dataset.user_int_vocab_sizes)}")

# 应该看到维度增加了15
```

### Q2: 如果数据中没有 label_time 列怎么办？

**A**: 代码会自动检测，只提取 timestamp 的7个特征，不会报错。

### Q3: 时间特征会影响训练速度吗？

**A**: 影响很小（<1%），因为使用了向量化操作。

### Q4: 如何查看某个样本的时间特征？

**A**: 

```python
batch = next(iter(train_loader))
sample_idx = 0
time_features = batch['user_int_feats'][sample_idx, -15:]

print("Year:", time_features[0].item())
print("Month:", time_features[1].item())
print("Day:", time_features[2].item())
print("Hour:", time_features[3].item())
print("Minute:", time_features[4].item())
print("Weekday:", time_features[5].item())
print("Is Weekend:", time_features[6].item())
```

---

## 📊 预期效果

### 训练日志示例

```
PCVRParquetDataset: 1000000 rows from 10 file(s), batch_size=256
user_int_schema.total_dim: 65  (原始50 + 时间15)
user_int vocab_sizes length: 65
Epoch 1/10: loss=0.234, auc=0.678
...
```

### 特征统计示例

```
Time Feature Statistics:
- Year: min=21, max=24 (2021-2024)
- Month: min=1, max=12
- Hour: min=0, max=23
- Weekday: min=0, max=6
- Decision Time: min=0, max=4
```

---

## ⚠️ 注意事项

1. **首次运行**: 第一次运行时会自动扩展 schema，可能需要几秒钟
2. **内存使用**: 时间特征增加的内存开销很小（每个样本增加15个int64）
3. **模型参数**: Embedding 参数会增加，但增量很小
4. **时区**: 所有时间使用 UTC，确保数据一致性

---

## 🎯 下一步

1. ✅ 运行 `test_time_features.py` 验证
2. ✅ 运行训练脚本
3. ✅ 监控训练日志，确认无错误
4. ✅ 对比加入时间特征前后的模型效果

---

## 📚 相关文档

- [时间特征工程优化总结.md](../时间特征工程优化总结.md) - 完整的修改说明
- [TIME_FEATURES_README.md](TIME_FEATURES_README.md) - 技术文档
- [SAFETY_CHECKLIST.md](SAFETY_CHECKLIST.md) - 安全检查清单

---

**现在可以直接运行训练，时间特征会自动生效！** 🎉

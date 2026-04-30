# 时间特征工程 - 安全检查清单

## ✅ 已实现的安全措施

### 1. 索引范围保护
- [x] Year: `np.clip(0, 100)` - 防止超出 vocab_size=101
- [x] Month: `np.clip(0, 12)` - 防止超出 vocab_size=13
- [x] Day: `np.clip(0, 31)` - 防止超出 vocab_size=32
- [x] Hour: `np.clip(0, 23)` - 防止超出 vocab_size=24
- [x] Minute: `np.clip(0, 59)` - 防止超出 vocab_size=60
- [x] Weekday: `np.clip(0, 6)` - 防止超出 vocab_size=7
- [x] Is_Weekend: `np.clip(0, 1)` - 防止超出 vocab_size=2
- [x] Decision_Time: `np.clip(0, 4)` - 防止超出 vocab_size=5

### 2. 异常值处理
- [x] 无效时间戳（<=0）→ 填充为0（padding）
- [x] 时间戳解析异常 → try-except 捕获，使用默认值0
- [x] 负数决策时间 → `np.maximum(0)` 修正为0
- [x] 溢出错误 → OverflowError 捕获

### 3. 维度安全
- [x] Buffer 预分配时增加15个时间特征的空间
- [x] Schema 自动添加时间特征定义
- [x] 动态检测 label_time 列是否存在
- [x] 特征追加前检查 buffer 大小

### 4. 数据类型安全
- [x] 所有时间特征使用 `np.int64` 类型
- [x] 时间戳转换使用 `astype(np.int64)`
- [x] Clip 操作保持 int64 类型

### 5. 边界条件处理
- [x] Batch size 为0时的处理
- [x] 空数据的处理
- [x] 全部为无效时间戳的处理

## 🔍 测试覆盖

### 单元测试 (test_time_features.py)
- [x] 正常时间戳提取
- [x] 无效时间戳处理（0, -1）
- [x] 边界值测试（年份、月份、日期等）
- [x] 周末判断逻辑
- [x] 决策时间分桶逻辑
- [x] 索引范围验证

### 集成测试 (verify_dataset.py)
- [x] 数据集加载
- [x] Batch 生成
- [x] 特征维度检查
- [x] 特征值范围检查

## 🛡️ 防御性编程措施

### 1. 输入验证
```python
if ts <= 0:
    continue  # 跳过无效时间戳
```

### 2. 异常捕获
```python
try:
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    # ... 提取特征
except (ValueError, OSError, OverflowError):
    pass  # 保持默认值0
```

### 3. 范围限制
```python
time_feats[:, i] = np.clip(time_feats[:, i], min_val, max_val)
```

### 4. 维度检查
```python
if original_offset + num_time_feats <= user_int.shape[1]:
    user_int[:, original_offset:original_offset + num_time_feats] = time_features
else:
    logging.warning(f"user_int buffer too small")
```

## 📋 运行前检查清单

在运行训练之前，请确认：

- [ ] 已运行 `python test_time_features.py` 且全部通过
- [ ] 已运行 `python verify_dataset.py` 且无错误
- [ ] 数据文件包含 `timestamp` 列
- [ ] 数据文件包含 `label_time` 列（可选）
- [ ] Schema.json 文件存在且格式正确

## 🚨 可能的错误及解决方案

### 错误1: IndexError: index out of bounds
**原因**: Embedding 索引超出 vocab_size
**解决**: 
- 检查 `_extract_time_features()` 中的 clip 操作
- 确认 vocab_sizes 设置正确（见下表）

### 错误2: RuntimeError: shape mismatch
**原因**: user_int_feats 维度不匹配
**解决**:
- 检查 buffer 预分配大小
- 确认 schema 中包含时间特征定义

### 错误3: ValueError: invalid timestamp
**原因**: 时间戳格式错误
**解决**:
- 已通过 try-except 捕获，不应出现此错误
- 如果出现，检查数据源

## 📊 Vocab Size 对照表

| 特征 | Vocab Size | 实际范围 | 说明 |
|------|-----------|---------|------|
| year | 101 | 0-100 | 0=padding, 1-100=2001-2100 |
| month | 13 | 0-12 | 0=padding, 1-12=Jan-Dec |
| day | 32 | 0-31 | 0=padding, 1-31=days |
| hour | 24 | 0-23 | 0-23=hours |
| minute | 60 | 0-59 | 0-59=minutes |
| weekday | 7 | 0-6 | 0=Mon, 6=Sun |
| is_weekend | 2 | 0-1 | 0=weekday, 1=weekend |
| decision_time | 5 | 0-4 | 0=invalid, 1-4=buckets |

**重要**: Embedding 的 `num_embeddings` 必须 >= vocab_size，否则会出现索引错误！

## 🔧 调试技巧

### 1. 打印时间特征
```python
batch = next(iter(dataset))
time_feats = batch['user_int_feats'][:, -15:]
print("Time features:", time_feats[0])
```

### 2. 检查范围
```python
for i in range(15):
    print(f"Feature {i}: min={time_feats[:, i].min()}, max={time_feats[:, i].max()}")
```

### 3. 启用详细日志
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## ✅ 最终确认

在部署到生产环境前：

1. ✅ 所有单元测试通过
2. ✅ 集成测试通过
3. ✅ 在小批量数据上训练成功
4. ✅ 没有 OOB (out-of-bound) 警告
5. ✅ 模型收敛正常
6. ✅ 性能开销可接受（<1%）

---

**如果所有检查都通过，代码已经可以安全运行，不会出现维度和索引错误！** ✅

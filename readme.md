# BP Relation Training README

这个文档只说明 `train_bp_relation_net.py` 的训练入口、参数含义和常用命令。

当前脚本训练的是同一个 relation model：

- `single_pick`
- `pair_pick`
- `ban`

输出文件默认是：

- `models/bp_relation_net.pt`
- `models/bp_relation_meta.json`
- `models/bp_relation_vocab.json`
- `models/bp_relation_split_manifest.json`

## 1. 最常用命令

首次训练：

```powershell
python train_bp_relation_net.py --from-scratch
```

常规增量 fine-tune：

```powershell
python train_bp_relation_net.py --fine-tune
```

重建 train/val split：

```powershell
python train_bp_relation_net.py --rebuild-split
```

偏向新数据的 fine-tune：

```powershell
python train_bp_relation_net.py --fine-tune --new-data-weight 4.0 --old-data-replay-ratio 0.35
```

更激进地只训本次新增数据：

```powershell
python train_bp_relation_net.py --fine-tune --fine-tune-new-only
```

偏 GPU 吞吐的训练：

```powershell
python train_bp_relation_net.py --fine-tune --batch-size 1024 --num-workers 4 --pin-memory --amp
```

## 2. 参数总览

脚本当前支持这些参数：

```text
--data-path
--model-path
--meta-path
--vocab-path
--split-manifest-path
--rebuild-split
--epochs
--batch-size
--lr
--negatives-per-positive
--ban-loss-weight
--val-ratio
--seed
--new-data-weight
--old-data-replay-ratio
--fine-tune-new-only
--device
--num-workers
--prefetch-factor
--grad-accum-steps
--log-interval
--fine-tune
--from-scratch
--compile
--pin-memory
--no-pin-memory
--persistent-workers
--no-persistent-workers
--amp
--no-amp
```

下面按用途解释。

## 3. 数据与输出路径

`--data-path`

- 训练数据路径。
- 默认：`data/bp_samples.sqlite`

`--model-path`

- 模型权重输出路径。
- 默认：`models/bp_relation_net.pt`

`--meta-path`

- 训练元信息输出路径。
- 默认：`models/bp_relation_meta.json`

`--vocab-path`

- 词表输出路径。
- 默认：`models/bp_relation_vocab.json`

`--split-manifest-path`

- 稳定 split manifest 路径。
- 默认：`models/bp_relation_split_manifest.json`
- 用来保证旧 group 的 train/val 划分不随每次训练改变。

## 4. Split 相关参数

`--val-ratio`

- 验证集 group 比例。
- 默认：`0.1`
- 按 group 划分，不是按单条样本随机切。

`--seed`

- 随机种子。
- 默认：`42`
- 会影响首次 split、新 group 分配、negative sampling 等。

`--rebuild-split`

- 强制重建全部 split manifest。
- 默认不启用。
- 只有在你明确想重新划 train/val 时才开。

说明：

- 正常情况下旧 group 会保持原 split。
- 新增数据只给新 group 分配 train/val。
- 不加 `--rebuild-split` 时，旧数据 split 不会变化。

## 5. 训练模式参数

`--fine-tune`

- 使用 fine-tune 模式。
- 这是当前默认模式。
- 如果现有 `model/vocab` 存在，会加载旧 checkpoint 继续训练。

`--from-scratch`

- 从零开始训练。
- 不加载旧 checkpoint。

`--fine-tune-new-only`

- fine-tune 时只用本次新增 train groups。
- 默认关闭。
- 如果本次没有新 group，会直接退出。

使用建议：

- 日常增量更新：`--fine-tune`
- 明确要重练全模型：`--from-scratch`
- 只想快速吸收新增数据：`--fine-tune --fine-tune-new-only`

## 6. 优化器与训练轮数

`--epochs`

- 训练 epoch 数。
- 默认：
  `from-scratch` 用脚本内默认 `10`
  `fine-tune` 用脚本内默认 `3`

`--lr`

- 学习率。
- 默认：
  `from-scratch` 为 `1e-3`
  `fine-tune` 为 `3e-4`

`--batch-size`

- DataLoader batch size。
- 默认：`512`

`--grad-accum-steps`

- 梯度累积步数。
- 默认：`1`
- 有效 batch size =
  `batch_size * grad_accum_steps`

示例：

```powershell
python train_bp_relation_net.py --batch-size 512 --grad-accum-steps 4
```

这等价于更大的有效 batch，但单步显存压力更小。

## 7. Pick / Ban 任务相关参数

`--negatives-per-positive`

- 每个正样本复制多少个训练 row。
- 默认：`4`
- 越大训练样本越多，吞吐压力也越高。

`--ban-loss-weight`

- 总训练里 ban loss 的权重。
- 默认：`0.8`
- 总体上等价于：

```text
total_loss = pick_loss + 0.8 * ban_loss
```

调参建议：

- pick 更重要：适当降低，例如 `0.5`
- ban 学得太弱：适当提高，例如 `1.0`

## 8. 增量 fine-tune 权重参数

这些参数只在 fine-tune 时最有意义。

`--new-data-weight`

- 新数据的采样放大系数。
- 默认：`4.0`
- 数值越大，训练越偏向本次新增 groups。

`--old-data-replay-ratio`

- 旧数据 replay 占比。
- 默认：`0.35`
- 数值越大，越保守，越不容易遗忘旧分布。

`--fine-tune-new-only`

- 不混旧数据，只训练新数据。
- 风险是遗忘旧分布更明显。

推荐理解：

- 想稳一点：`--new-data-weight 2.0 --old-data-replay-ratio 0.5`
- 想更快吸收新 meta：`--new-data-weight 4.0 --old-data-replay-ratio 0.35`
- 想极端偏新数据：`--fine-tune-new-only`

## 9. GPU / DataLoader 参数

`--device`

- 指定设备。
- 默认自动选：
  `cuda` 如果可用，否则 `cpu`

示例：

```powershell
python train_bp_relation_net.py --device cuda
python train_bp_relation_net.py --device cpu
```

`--num-workers`

- DataLoader worker 数。
- 默认：`4`

`--prefetch-factor`

- 每个 worker 的预取批次数。
- 默认：`4`
- 仅 `num_workers > 0` 时生效。

`--pin-memory`

- 开启 pinned memory。
- 默认开启。
- 只在 CUDA 下有意义。

`--no-pin-memory`

- 关闭 pinned memory。

`--persistent-workers`

- 让 DataLoader workers 跨 epoch 常驻。
- 默认开启。

`--no-persistent-workers`

- 关闭 persistent workers。

`--amp`

- 开启混合精度训练。
- 默认开启。

`--no-amp`

- 关闭 AMP。

`--compile`

- 尝试 `torch.compile(model)`。
- 默认关闭。
- 有些环境会更快，有些环境编译开销大或者不稳定。

推荐 GPU 吞吐配置：

```powershell
python train_bp_relation_net.py --fine-tune --batch-size 1024 --num-workers 4 --pin-memory --persistent-workers --amp
```

如果显存不足：

```powershell
python train_bp_relation_net.py --fine-tune --batch-size 512 --grad-accum-steps 2 --amp
```

## 10. 日志参数

`--log-interval`

- 训练进度条更新间隔。
- 默认：`50`

它影响的是训练中间 loss 刷新频率，不影响最终指标。

## 11. 常见训练场景

场景 1：第一次完整训练

```powershell
python train_bp_relation_net.py --from-scratch
```

场景 2：数据刚更新，正常继续训练

```powershell
python train_bp_relation_net.py --fine-tune
```

场景 3：新数据很重要，希望更偏向新数据

```powershell
python train_bp_relation_net.py --fine-tune --new-data-weight 4.0 --old-data-replay-ratio 0.35
```

场景 4：只训新增数据，追求更新速度

```powershell
python train_bp_relation_net.py --fine-tune --fine-tune-new-only
```

场景 5：怀疑旧 split 不合理，强制重划

```powershell
python train_bp_relation_net.py --rebuild-split
```

场景 6：GPU 还没吃满，继续加吞吐

```powershell
python train_bp_relation_net.py --fine-tune --batch-size 1024 --num-workers 8 --prefetch-factor 4 --amp --pin-memory
```

## 12. 训练日志怎么看

启动时常见日志：

- `split manifest loaded/created`
  当前是读取旧 split 还是创建新 split
- `old groups kept`
  旧 group 保持原 split 的数量
- `new groups assigned`
  本次新分配 split 的 group 数量
- `Fine-tune mode`
  当前是否按增量模式训练
- `New train groups / Old train groups`
  新旧训练 group 数量
- `Device / GPU / AMP enabled`
  当前硬件与精度模式

每个 epoch 常见日志：

- `pick train loss / pick val loss`
- `pick pairwise accuracy`
- `ban train loss / ban val loss`
- `ban pairwise accuracy`
- `ban top-1 accuracy / ban top-3 accuracy`
- `negative refresh time`
  epoch 开头预生成 hard negatives 的耗时
- `train samples/sec`
  训练阶段吞吐
- `epoch time`
  整个 epoch 总耗时
- `peak_gpu_memory_mb`
  峰值显存

## 13. 建议的排查顺序

如果训练太慢，优先看：

1. `negative refresh time`
2. `train samples/sec`
3. `num_workers`
4. `batch_size`
5. `grad_accum_steps`
6. `peak_gpu_memory_mb`

判断逻辑：

- `negative refresh time` 很高：瓶颈在 epoch 前负采样准备
- `train samples/sec` 很低但显存不高：通常 batch 太小或 DataLoader 太弱
- `peak_gpu_memory_mb` 很低且 GPU 利用率低：通常喂数不够快
- `peak_gpu_memory_mb` 很高：先减小 `batch-size`，必要时加 `grad-accum-steps`

## 14. 一条实用命令

如果你现在只想“正常增量训练，尽量用 GPU，偏向新数据”，可以直接用：

```powershell
python train_bp_relation_net.py --fine-tune --new-data-weight 4.0 --old-data-replay-ratio 0.35 --batch-size 1024 --num-workers 4 --pin-memory --persistent-workers --amp
```

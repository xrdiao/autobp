# AutoBP

AutoBP 是《Summoners War》世界竞技场（RTA）的客观 BP 决策工具。它把历史 BP、公开统计和技能机制整理成一个事件级模型，用于推荐我方下一手、预测对方下一手和推荐 Ban。

账号信息只提供我方拥有的魔灵，作为候选合法性掩码；Box、符文和账号战绩不参与模型排序。一次双选中的两个魔灵是无序集合，双方事件的先后顺序仍由 BP 时间线保留。

作者：[xrdiao](https://github.com/xrdiao/LSTP-Nav/commits?author=xrdiao)

## 功能

- 专用推荐模型：Relation 负责选人排序，Prefix Value 负责局面价值，Full Draft Value 负责 Ban 与完整阵容评估。
- 客观证据：融合 Lucksack、SWKit、SWRT 和 SWARFARM 的公开统计、对位与机制标签。
- 自动动作大小：根据当前 BP 轮次自动生成单个或两个魔灵的动作。
- 有限深度搜索：使用 Prefix Value 价值模型评估候选动作和对手响应。
- Web 与桌面端：支持手动点选、Box 管理、截图识别和推荐结果应用。
- 回放评测：固定随机种子抽取历史前缀，统计 Top-k 命中率、价值差、校准误差和延迟。
- Steam 安全：截图识别只处理用户主动提供的截图，不枚举 Steam 或游戏进程，不读取窗口句柄、内存或图形 Hook。

推荐链路如下：

```text
BP 事件时间线
    ↓
Relation（单选 / 双选 / 对手） + Prefix Value（局面价值） + Draft Value（Ban）
    ↓
公开统计与技能机制证据
    ↓
有限深度响应搜索
    ↓
可执行的 BP 动作
```

## 安装

克隆仓库并进入项目目录。

```bash
git clone https://github.com/xrdiao/autobp.git
cd autobp
```

根据项目环境文件创建 Conda 环境。

```bash
conda env create -f environment.yml
```

激活环境。

```bash
conda activate autobp
```

安装截图识别需要的 Chromium 运行时。

```bash
python -m playwright install chromium
```

确认 Python 和 PyTorch 来自 autobp 环境。

```bash
python -c "import sys, torch; print(sys.executable); print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

`environment.yml` 固定 Python 3.11，并包含 `torch==2.14.0`、Streamlit、OpenCV、Pillow 和 Playwright。

## 快速使用

首次使用先同步魔灵目录、别称、图片地址和公开统计。

```bash
python build_dataset/sync_lucksack_data.py --combo-limit 2000 --group-concurrency 2
```

同步 SWKit 别称和 RTA 评级。

```bash
python build_dataset/sync_swkit_data.py
```

同步 SWRT 对位数据和 SWARFARM 技能机制。

```bash
python build_dataset/sync_swrt_data.py --monster-limit 400 --max-workers 4
python build_dataset/sync_swarfarm_mechanisms.py
```

根据公开目录补全别称、图片、属性和系列映射。

```bash
python -m build_dataset.generate_alias_and_images
```

启动 Web 推荐器。

```bash
streamlit run app_web.py --server.address 0.0.0.0 --server.port 8501
```

浏览器打开本机页面。

```text
http://127.0.0.1:8501
```

启动桌面版。

```bash
python -m apps.desktop
```

完成训练后，Relation、Prefix Value 和 Draft Value 的模型、元数据与词表会写入 `models/`，Web 和桌面端读取这三组专用模型。BP 状态仍使用统一的事件时间线表达。重建样本后必须重新训练 Draft Value，双方 Ban 字段才会进入模型。

## 数据

同步公开统计并从已有比赛 HTML 构建事件级样本；首次获取比赛页面时加上 `--download`。

```bash
python build_dataset/run_all.py
```

```bash
python build_dataset/run_all.py --download
```

主要数据源如下：

| 数据 | 用途 |
| --- | --- |
| [Lucksack](https://lucksack.gg/) | 赛季、段位、BP 统计、组合统计、魔灵目录和图片 |
| SWKit | 中文别称、合作角色别名和 RTA 评级 |
| [SWRT](https://m.swranking.com/) | RTA 统计和对位 |
| [SWARFARM](https://swarfarm.com/) | 技能效果和机制标签 |

运行时生成的公开快照位于 `json/`，历史比赛数据位于 `data/` 或 `dataset/`。这些目录被 `.gitignore` 忽略，重新克隆后需要由使用者重新同步或提供自己的数据。

生成的关键文件包括：

- `json/lucksack_monster_catalog.json`：可用魔灵、属性、系列和图片文件名。
- `json/monster_alias_map.json`：别称到标准名称的映射。
- `json/monster_images.json`：标准名称到图片地址的映射。
- `json/lucksack_stats.json`、`json/swrt_rta_stats.json`：公开统计快照。
- `json/sw_mechanism_profiles.json`：技能机制与威胁标签。
- `data/bp_samples.sqlite`：逐回合 BP 训练和评测样本。

样本只使用事件级字段：`events` 是当前动作之前的完整 BP 时间线，`next_action_slugs` 是下一件单选或无序双选动作，`full_events` 用于完整阵容和 Ban 标签。旧的扁平选人样本不会被训练器兼容；切换到当前格式时请强制重建样本库。

从原始比赛重新生成事件级样本。

```bash
python build_dataset/run_all.py --force-rebuild
```

如果排行榜正在维护，跳过在线同步，直接使用本地 `dataset/` 中已有的比赛页面。

```bash
python build_dataset/run_all.py --skip-swkit --skip-lucksack --skip-swrt --skip-mechanisms --skip-images --force-rebuild
```

排行榜恢复后，使用固定版本和已有排行榜清单继续下载。

```bash
python build_dataset/run_all.py --download --dataset-version v3.0 --use-existing-version-csv --force-rebuild
```

目录同步时会过滤不可用的未觉醒条目；二次觉醒条目只有在目录明确标记为可搜索时才会保留。

## 训练

在样本库已生成后，同步公开数据、审计事件级样本并训练三组专用模型。

```bash
python -m training.pipeline --sync-web-data
```

只重新审计和训练已经生成的本地样本。

```bash
python -m training.pipeline
```

只使用本地数据训练 Relation、Prefix Value 和 Draft Value。

```bash
python -m training.pipeline --skip-audit --relation-epochs 3 --prefix-epochs 10
```

在 CPU 上训练并把进度输出保留在终端。

```bash
python -m training.pipeline --skip-audit --device cpu
```

训练产物如下：

```text
models/bp_relation_net_v2.pt
models/bp_relation_meta_v2.json
models/bp_relation_vocab_v2.json
models/bp_prefix_value_net.pt
models/bp_prefix_value_meta.json
models/bp_prefix_value_vocab.json
models/bp_draft_value_net.pt
models/bp_draft_value_meta.json
models/bp_draft_value_vocab.json
```

Relation 训练器保留双方选人顺序，同时把同一双选事件编码为无序 pair；Prefix Value 按事件前缀训练并进行概率校准；Draft Value 使用完整五选四阵容和 Ban 监督。

常用参数：

- `--epochs`：训练轮数。
- `--batch-size`：批大小。
- `--negatives-per-positive`：Relation 每个正样本的负样本数量。
- `--val-ratio`：按比赛分组的验证集比例。
- `--max-samples`：调试时限制样本数，默认使用全部数据。
- `--device cuda|cpu`：选择训练设备。

## 评测

先检查样本的时间线、选人数量和下一手字段。

```bash
python -m evaluation.audit_prefix_data --data-path data/bp_samples.sqlite
```

从验证集按固定种子随机抽取 2000 个事件前缀进行回放评测。

```bash
python -m evaluation.replay --eval-split val --limit 2000 --top-k 6 --seed 42
```

评估当前划分中的全部有效前缀。

```bash
python -m evaluation.replay --eval-split val --limit 0 --top-k 6 --seed 42
```

把完整评测结果保存为 JSON。

```bash
python -m evaluation.replay --limit 2000 --top-k 6 --seed 42 --output evaluation/legacy_replay.json
```

评测输出包括：

- `top1_hit_rate`、`top3_hit_rate`：历史下一手是否出现在推荐前 k 名。
- `mean_recommended_value`、`mean_actual_value`、`mean_value_lift`：Prefix Value 对推荐动作和历史动作的比较。
- `brier_score`、`log_loss`、`ece`：按比赛抽取结果样本后的概率校准指标。
- `mean_latency_ms`：平均推荐延迟。

行为命中率表示模型复现历史选择的程度，不等同于反事实胜率；比较版本时应同时观察价值差和校准指标。

## 使用说明

Web 页面中，当前对方选人数量决定预测一个还是两个魔灵。点击推荐卡片的应用按钮会把整件动作写入当前 BP 状态；双选成员在显示和缓存中按无序 pair 处理，事件顺序不会被交换。

“我的 Box”只限制我方候选。关闭 Box 限制后，Relation 可以从完整目录和统计候选中推荐；对方预测始终不使用账号 Box。

账号 Box 会保存到本地 `json/my_box.json`。可选的 `json/account_profile.json` 只用于初始化拥有列表：

```json
{
  "schema_version": 1,
  "source": "manual_normalized",
  "owned_monsters": ["Monster Name"]
}
```

桌面截图识别只处理用户主动选择的图片。批量识别截图：

```bash
python sw_ui_detection/run_bp_screen.py
```

识别单张截图并输出调试图：

```bash
python -m sw_ui_detection --image path/to/bp.png --json-out sw_ui_detection/output/bp.json --debug-out sw_ui_detection/output/bp_preview.png
```

## 项目结构

```text
app_web.py                 Streamlit 入口
apps/desktop.py            Tk 桌面入口
bp_core/                   BP 事件、状态、词表、数据和模型公共组件
bp_recommender/relation.py  Relation 选人与对手预测
bp_recommender/prefix_value.py  Prefix Value 局面评估
bp_recommender/ban_value.py  Draft Value Ban 评估
bp_recommender/decision_engine.py  统一状态上的有限深度响应搜索
training/relation.py       Relation 训练入口
training/prefix_value.py   Prefix Value 训练入口
training/draft_value.py    Draft Value 训练入口
training/pipeline.py       数据同步、审计和三模型训练流水线
training/data_pipeline.py  样本读取与动作标签构造
evaluation/                回放评测和数据审计
build_dataset/             公开数据同步与目录生成
sw_ui_detection/           截图定位和魔灵识别
models/                    三组专用模型训练产物
tests/                     单元测试
```

运行模块时保持项目根目录为当前目录：

```bash
python -m training.pipeline
python -m evaluation.replay --limit 2000
```

## 限制

公开数据没有逐帧战斗轨迹、符文速度、行动条、目标选择、技能冷却和按键记录，因此模型输出是概率排序和局面估计，不是必胜结论。若要训练战斗内策略，需要额外采集这些轨迹数据。

# BP Training README

当前 BP 决策链由三个职责清晰的部分组成：

- `relation v2`：根据真实时间前缀（chronological prefix）学习选人 / 双选兼容性，使用选人顺序、双方侧别、当前回合和 action-state cross attention。
- `prefix value`：根据未完成 BP 状态预测我方最终胜率，训练数据只来自逐回合前缀，不把完整阵容模型直接用于半成品状态。
- 完整 5v5 后的 Ban：继续使用 `bp_draft_value_net.pt`，因为此时输入已经是完整阵容。
- 双选采用 Relation 先筛选、再对少量高分组合计算完整统计和机制证据，避免每个组合重复扫描全部外部数据。

桌面端排序逻辑：

- 我方选人：先由 Relation v2、Lucksack、RTA 和机制层生成候选，再对前 12 个候选做“我方动作 -> 对方前三个反制 -> 我方状态价值”的两步预演，最终按 `0.60 * adversarial_value + 0.15 * immediate_value + 0.15 * relation + 0.10 * evidence` 排序。
- 对方选人预测：模拟对方动作后的我方前三个最佳回应，按我方回应后剩余胜率反向排序。
- Lucksack 统计层融合当前赛季 / 补丁的单体胜率、双魔灵 / 三魔灵组合胜率，以及当前 BP 阶段的 `pick_rate / win_rate`；样本数会参与置信度收缩，避免低样本组合直接主导排序。
- 网页统计还会提供 `pick_win_rate / ban_rate / lead_rate`、先后手总胜率、段位分布和数据覆盖度；Ban 推荐使用 Ban 压力和目标胜率作为额外证据。
- SWKit 的 RTA 评级榜单作为小权重先验加入统计层，用于补充低出场或新版本样本不足的魔灵。
- SWRT 对位层会加入候选对敌方单体 / 双体阵容的实际胜率；Ban 时也会用目标对我方阵容的对位胜率作为小权重证据。
- SWARFARM 技能机制层在存在敌方阵容时参与排序；Relation 候选预筛选也会使用机制分，避免双选时把低热度反制位提前过滤掉。
- Ban：完整 5v5 后先比较 Ban 前后的完整阵容 value，再融合机制威胁削弱和网页 Ban 压力。

## Lucksack 实时数据

根页面 `https://lucksack.gg/` 会动态加载当前赛季、补丁、段位分组、魔灵目录，以及单体 / 双体 / 三体统计。项目通过浏览器上下文读取这些公开接口，避免直接请求 API 时被站点的访问策略拦截。

当前同步使用的接口：

- `https://api.lucksack.gg/seasons`：当前赛季。
- `https://api.lucksack.gg/seasons/{season}/rank-groups`：段位分组和对应 rank id。
- `https://api.lucksack.gg/seasons/{season}/patches`：补丁区间。
- `https://api.lucksack.gg/statistics/ranks`：按日期记录的段位分布。
- `https://api.lucksack.gg/statistics/solo`：单魔灵出场数、使用率、Ban 率、胜率。
- `https://api.lucksack.gg/statistics/duo` / `trio`：双魔灵、三魔灵组合胜率。
- `https://api.lucksack.gg/meta-dashboard/summary`：总对局、先后手胜率、最高胜率和最高 Ban。
- `https://api.lucksack.gg/meta-dashboard/draft-position`：六个 BP 阶段的出场数、使用率和胜率。
- `https://static.lucksack.gg/data/monsters_catalog.json`：魔灵目录和图片文件名。

## SWRT 公开数据

SWRT 的 H5 / 小程序共用公开接口 `https://m.swranking.com/api`。当前同步使用：

- `/monster/statistical`：全量魔灵的出场、胜率、Ban 率和先手率。
- `/monster/highdata`：指定魔灵的高胜 / 低胜 1v1 对位，以及少量双魔灵对位。

这些数据不写入逐回合训练样本，只作为当前版本的统计和对位证据。同步默认抓取出场量最高的 400 个魔灵的 `highdata`，避免对公开接口产生不必要的请求：

```bash
python build_dataset/sync_swrt_data.py --monster-limit 400
```

输出文件：

- `json/swrt_rta_stats.json`：SWRT 统计、1v1 和一对二对位快照。

推荐器会在有对应记录时融合 `matchup_score`；没有记录时仍使用 Relation、Lucksack、SWKit 和机制层，不会把缺失对位当成负面证据。

手动同步：

```bash
conda activate autobp
python build_dataset/sync_lucksack_data.py --combo-limit 2000
```

同步文件：

- `json/lucksack_stats.json`：当前赛季和补丁的分段统计快照。
- `json/lucksack_monster_catalog.json`：站点目录，包含新魔灵名称、属性和图片文件名；可搜索的四星及以上目录项会作为新候选补充。
- `json/swkit_monster_aliases.json`：SWKit 中文别称和合作角色别名。
- `json/swkit_rta_tiers.json`：SWKit RTA 评级榜单快照。
- `json/swrt_rta_stats.json`：SWRT 当前统计和对位数据快照。
- `json/sw_mechanism_profiles.json`：根据 SWARFARM 技能效果自动生成的机制标签、威胁标签、基础属性、技能冷却、倍率和属性缩放信息。

完整数据流水线默认先同步 Lucksack，因此换赛季、换补丁或目录出现新魔灵后，下一次运行会自动更新候选池：

```bash
python build_dataset/run_all.py
```

流水线还会同步 SWARFARM 的魔灵技能和技能效果，用于识别控制、行动条、冷却、续航、复活、炸弹/持续伤害和防御增益等机制。推荐时会比较候选加入前后的敌方威胁覆盖变化，并在推荐原因中显示命中的反制机制。

单独同步 SWKit：

```bash
python build_dataset/sync_swkit_data.py
```

单独同步机制数据：

```bash
python build_dataset/sync_swarfarm_mechanisms.py
```

## 一键同步并重训推荐器

当网页数据或本地比赛数据已经到齐时，直接运行：

```bash
conda activate autobp
python train_all_models.py --sync-web-data
```

这条命令会依次同步 Lucksack / SWKit / SWRT / SWARFARM，审计逐回合数据，然后增量训练 relation v2、prefix value 和完整阵容 Ban 模型。未来出现新魔灵时，网页目录、别称、RTA 评级、SWRT 当前统计和机制标签会先进入推荐器；即使新魔灵暂时没有历史训练样本，也会通过当前网页统计和机制层参与候选排序。

只重训本地数据，不重新抓网页：

```bash
python train_all_models.py
```

需要完全重训 relation v2 时显式指定：

```bash
python train_all_models.py --sync-web-data --from-scratch
```

常用控制参数：

- `--device cuda` / `--device cpu`：指定训练设备。
- `--skip-relation`、`--skip-prefix`、`--skip-draft`：跳过对应模型。
- `--skip-audit`：确认数据已审计后跳过数据检查。
- `--relation-epochs`、`--prefix-epochs`：覆盖对应训练轮数。

只处理本地比赛数据时可以跳过线上同步：

```bash
python build_dataset/run_all.py --skip-lucksack --skip-swkit --skip-mechanisms
```

默认使用 `102`（Guardian 全段位）统计。同步会保存所有可用段位分组；推荐器读取时可用环境变量切换：

```bash
BP_LUCKSACK_RANK_GROUP=16 streamlit run app_web.py --server.address 0.0.0.0 --server.port 8501
```

当前统计层负责整体强度、我方阵容协同、BP 阶段和 RTA 评级先验；机制层额外分析敌方阵容威胁以及候选的反制覆盖。机制层是可解释的辅助评分，不会覆盖 Relation v2 的历史数据排序，也不会把单个技能标签当成绝对硬规则。

### 符文速度、技能循环和临场操作

当前推荐器可以直接使用魔灵基础速度、队长速度、技能冷却、技能效果、命中次数和倍率缩放信息，因而能识别“速度/行动条/冷却控制”等机制类型。但 SWARFARM 和 SWRT 的公开接口没有提供每场 RTA 的实际符文速度、速度调优、符文套装、实时行动条、技能剩余冷却或玩家按键记录，所以当前不能凭这些网页数据可靠还原具体速度轴和临场操作。

要把这三项能力做实，需要继续接入两类数据：

- 符文构筑：每个 Box 魔灵的实际速度、命中、抵抗、暴击和套装，以及队长技能后的有效速度。
- 战斗轨迹：逐帧行动顺序、攻击目标、技能选择、行动条、Buff / Debuff、剩余冷却、复活和胜负结果。

前者可作为速度阈值和速度轴特征，后者可训练技能选择 / 目标选择策略（policy），再通过现有截图识别模块把战斗状态送入实时推荐。没有这两类数据时，系统会保持统计和机制推荐，不会把基础速度猜成实际符文速度。

数据处理性能：

- HTML 抽取、怪物目录和图片索引优先使用 `lxml`；未安装时自动使用 `html.parser`。
- SQLite 记录采用批量 `executemany` 写入，并一次查询待更新 key，避免每条记录执行存在性查询。
- BP 展开阶段复用增量前缀和后缀列表，减少重复扫描。
- 名称归一化和 dataset version 解析使用进程内缓存。
- 不使用 `--export-pt` 时，数据抽取模块不需要加载 PyTorch。
- HTML 抽取默认最多使用 8 个进程，可用 `--extract-max-workers` 覆盖。

## 0. Conda 环境

项目使用独立的 Conda 环境 `autobp`。不要在系统 Python 或系统 pip 中安装项目依赖。

首次创建环境：

```bash
cd /home/rong/backup/autobp
conda env create -f environment.yml
conda activate autobp
```

环境文件更新后：

```bash
conda activate autobp
conda env update -f environment.yml
```

确认当前命令使用的是 Conda 环境：

```bash
conda activate autobp
python -c "import sys; print(sys.executable)"
```

输出路径应位于 `.../.conda/envs/autobp/` 或 Anaconda 的 `envs/autobp/` 下。

PyTorch 也必须安装在已经激活的 `autobp` 环境中。根据机器的 CPU、CUDA 或 ROCm 情况，使用 PyTorch 官方安装页面生成命令，然后将命令中的 `pip` 替换为：

```bash
python -m pip
```

例如 CPU 环境：

```bash
conda activate autobp
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

安装浏览器运行时：

```bash
conda activate autobp
python -m playwright install chromium
```

注意：环境文件使用的是 Python 版 `playwright-python`，不要单独安装同名的 Node.js 包 `playwright`。如果环境是按旧版环境文件创建的，请修复：

```bash
conda activate autobp
conda remove -y playwright
conda install -y -c conda-forge playwright-python
python -c "import playwright; print(playwright.__file__)"
python -m playwright install chromium
```

验证环境：

```bash
conda activate autobp
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

## 1. 推荐执行顺序

先检查展开后的逐回合数据：

```bash
python audit_bp_prefix_data.py --data-path data/bp_samples.sqlite
```

训练 relation v2：

```bash
python train_bp_relation_net.py --from-scratch
```

训练 prefix value：

```bash
python train_prefix_value_net.py
```

Linux / Windows Web UI：

```bash
streamlit run app_web.py --server.address 0.0.0.0 --server.port 8501
```

浏览器访问 `http://127.0.0.1:8501`。如果从另一台机器访问，使用运行主机的局域网 IP 和端口 `8501`。
进入页面后在左侧魔灵图库直接点击图标填入当前 BP 轮次，右侧点击已选图标可以移除；`我的 Box` 也使用图标切换。
BP 图标选人使用浏览器局部组件，点击时不会重新加载整个页面；搜索、筛选和翻页在同一局部区域内完成。

relation v2 输出：

- `models/bp_relation_net_v2.pt`
- `models/bp_relation_meta_v2.json`
- `models/bp_relation_vocab_v2.json`

prefix value 输出：

- `models/bp_prefix_value_net.pt`
- `models/bp_prefix_value_meta.json`
- `models/bp_prefix_value_vocab.json`

## 2. Relation 训练

常规增量 fine-tune：

```bash
python train_bp_relation_net.py --fine-tune
```

偏向新数据：

```bash
python train_bp_relation_net.py --fine-tune \
  --new-data-weight 4.0 \
  --old-data-replay-ratio 0.35
```

只使用本次新增 group：

```bash
python train_bp_relation_net.py --fine-tune --fine-tune-new-only
```

默认每个 epoch 会对部分 train context 做 model-aware hard negative mining。关闭它：

```bash
python train_bp_relation_net.py --hard-negative-ratio 0
```

主要参数：

- `--data-path`：默认 `data/bp_samples.sqlite`
- `--model-path`：默认 `models/bp_relation_net_v2.pt`
- `--meta-path`：默认 `models/bp_relation_meta_v2.json`
- `--vocab-path`：默认 `models/bp_relation_vocab_v2.json`
- `--epochs`：从零训练默认 `10`，v2 fine-tune 默认 `3`
- `--batch-size`：默认 `1024`
- `--negatives-per-positive`：默认 `4`
- `--hard-negative-ratio`：默认 `0.25`
- `--hard-negative-candidate-pool`：默认 `64`
- `--pick-eval-candidate-cap`：默认 `96`，设为 `0` 做完整候选评估
- `--pick-eval-max-samples`：默认 `2048`，控制每轮 top-k 评估的验证样本数
- `--ban-loss-weight`：默认 `0.8`
- `--val-ratio`：默认 `0.1`
- `--seed`：默认 `42`
- `--new-data-weight`：默认 `4.0`
- `--old-data-replay-ratio`：默认 `0.35`
- `--device`：默认自动选择 CUDA / CPU
- `--num-workers`：默认 `4`
- `--grad-accum-steps`：默认 `1`
- `--amp` / `--no-amp`
- `--compile`

训练日志中的关键指标：

- `single top-1 / top-3`：单选真实候选在候选集中的命中率
- `pair top-1 / top-3`：双选真实候选在候选集中的命中率
- `ban top-1 / top-3`：Ban 候选命中率
- `pairwise accuracy`：只反映正负样本相对排序，不能代替 top-k 指标
- `negative refresh time`：hard negative 准备耗时

## 3. Prefix Value 训练

`train_prefix_value_net.py` 使用 `data/bp_samples.sqlite` 中每场比赛的逐回合样本：

- 按 `match_id` 划分 train / validation，避免同一比赛的不同 prefix 泄漏到两侧。
- 标签是该场比赛最终结果。
- 训练时屏蔽最终 Ban 字段，避免选人阶段读取未来信息。
- 使用 `turn`、`first_pick_side`、`current_side` 三个数值特征。
- 队长字段作为静态上下文；模型输出在验证集上做 temperature calibration。

常用命令：

```bash
python train_prefix_value_net.py --epochs 10 --batch-size 256
```

## 4. 数据排查

`audit_bp_prefix_data.py` 会检查：

- `current_side` 是否与 `turn` 和 `first_pick_side` 一致
- 已选数量是否等于 `turn - 1`
- `next_pick` 是否已错误地出现在前缀中
- 是否缺少 `next_pick`

如果审计失败，应先修复数据展开，再训练模型。

## 5. 参考思路

实现只吸收通用方法，不直接套用其他游戏规则：

- OpenSpiel：把 BP 视为有明确状态、动作和合法动作集合的 sequential game。
- The Art of Drafting：把 drafting 建模为顺序决策，强调候选动作价值，而不是只拟合出场频率。
- `lightd22/swainBot`：参考实际游戏 Bot 按 draft 阶段组织状态和候选动作的工程方式。

参考链接：

- https://github.com/google-deepmind/open_spiel
- https://arxiv.org/abs/2303.11820
- https://github.com/lightd22/swainBot

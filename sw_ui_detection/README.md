# SW UI Detection

这个目录现在分成两层：

- 通用层
  - [monster_bank.py](monster_bank.py)
  - [detector.py](detector.py)
  - 用来做“这张图里有没有像魔灵头像框的区域”的基础检测
- BP 专用层
  - [bp_screen_analyzer.py](bp_screen_analyzer.py)
  - 用来按世界竞技场 BP 界面的固定布局，直接切出：
    - 我方 5 个选人框
    - 对方 5 个选人框
    - 当前轮到哪一边选
    - 我方下方 box 区域

## 为什么先用截图识别

这是故意的第一步，不是最终形态。

原因很简单：

1. 先把“识别逻辑”在静态截图上跑通，最容易验证。
2. 一旦静态截图识别稳定了，后面只需要把“图片来源”换成窗口截图、模拟器截图或实时录屏帧，后面的布局识别和魔灵匹配都不用重写。
3. 你后面要做“识别对方已选魔灵”，本质上还是：
   - 先拿到一张当前画面
   - 再在固定区域里做头像匹配

所以截图识别是底层能力，不是绕路。

## 直接用 Python 启动

如果你不想走 CLI，请在仓库根目录直接运行：

```powershell
conda activate autobp
python sw_ui_detection/run_bp_screen.py
```

这个脚本会自动：

- 读取 `sw_ui_detection/input/` 里的图片
- 对每张图做 BP 布局分析
- 把结果写到 `sw_ui_detection/output/`

输出包括：

- `xxx_bp.json`
- `xxx_bp_preview.png`

桌面版里的 `Capture Current Screen` 使用的是用户点击后触发的系统截图，再由图像内容定位游戏画布。它不会枚举 Steam / 游戏进程，不读取窗口句柄，不访问游戏内存，也不做图形 API Hook。识别失败时仍可以继续用剪贴板截图方式。

## 放图方式

把你的 BP 截图放进：

```text
sw_ui_detection/input/
```

然后执行：

```powershell
python sw_ui_detection/run_bp_screen.py
```

## 也可以直接在 Python 代码里调用

```python
from sw_ui_detection import BPDraftScreenAnalyzer

analyzer = BPDraftScreenAnalyzer()
result = analyzer.analyze("sw_ui_detection/input/example.png")
print(result.to_dict())
```

## 当前 BP 专用分析器能输出什么

- `is_bp_screen`
  - 这张图是不是大概率是 RTA BP 界面
- `confidence`
  - BP 界面置信度
- `current_pick_side`
  - 当前更像轮到 `my`、`enemy` 还是 `unknown`
- `my_slots`
  - 我方 5 个选人框位置
- `enemy_slots`
  - 对方 5 个选人框位置
- `my_box_region`
  - 我方下方 box 区域

## 下一步适合做什么

- 继续收集不同分辨率、缩放和模拟器边框样本，用回归报告微调画布定位
- 对 `my_box_region` 里的 box 图标做网格切分
- 只在用户主动触发时抓取当前屏幕，避免引入进程枚举、窗口句柄读取或 Hook 类截图路径

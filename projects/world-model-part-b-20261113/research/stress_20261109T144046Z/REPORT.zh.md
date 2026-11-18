# 视频世界模型研究接续：方案清理、抽帧修复与尺度压力测试

日期：2026-11-09。用户明确授权删除气象方案、继续视频研究。

## 研究问题与已有进度

本轮研究：冻结TinyUNet的合成优势是否对目标尺寸和输入画幅变化稳健？同时解决Part B抽帧中已发现的颜色失真，保证后续裁判输入可用。

当前研究对象为Wan2.1 1.3B与CogVideoX-5B的生成视频。已有冻结Part A：2模型×4提示×3生成种子=24视频、1560帧；0/24通过完整物理测量。原实验结果继续保留，不把测量失败解释为物理错误。上一轮120张匿名人工标注帧尚无人工标签。

已读取 `shared/research_plans/world_model_video_continuation_20261109T143307Z.md`，本轮方法记录于 `shared/research_plans/world_model_video_cleanup_stress_20261109T144046Z.md`。本次未开展新文献检索；历史设计中的新颖性声明不作为本轮已核实结论。

## 气象方案清理与代码定位

在当前检出 `/data/AI_Research/projects/world-model-physical-consistency` 中，删除旧 `proposal.md`、`proposal.zh.md`、`experiment-report.md`、`experiment-report.zh.md`，用视频研究入口替换旧README。

另对本机原始main和video-bench工作树执行相同清理，并移除原始main下未提交的 `docs/superpowers/specs/2026-10-15-pilot-design.md`。三份检出共退役16份文档（包括3份旧README），均事先在仓库外归档。恢复档案位于 `/data/AI_Research/shared/handoff/world_model_cleanup_20261109T144046Z/`；文件哈希、其他检出删除清单和代码导入清单一并保留。Git历史未重写。

当前检出导入原video-bench的39份实际源代码/测试/视频计划文件，以SHA256记录来源；原工作树有未提交更改，因此不能只靠commit号重现。本轮帧采样修复位于当前检出；原工作树继续作为旧视频、权重、冻结结果的来源。此处没有复制大模型权重或重新生成视频。未向远端提交、推送或公开发布。

## 抽帧问题：根因与验证

采用实际编码的全红/全蓝控制视频，核对保存的单帧与拼图中心像素。修复前，单帧颜色正确而拼图红蓝互换：imageio输出RGB，拼图直接传给期望BGR的OpenCV PNG写入器，造成通道交换。

代码在拼图单元写入前显式转换RGB→BGR。另发现单帧视频或请求1帧会因采样公式除以零；增加单帧分支，并明确拒绝非正采样数量。

`tests/test_frame_fidelity.py` 的6个真实视频回归用例在修复前全部失败（2个颜色、2个单帧、2个非法数量），修复后与原项目套件合并，实际结果 **48 passed in 7.87s**。修复不表示Part B裁判已经校准；固定版本裁判、实际控制画面人工核实和负例检验仍待进行。

## 压力测试设计与实验单位

所有压力测试数据均为程序合成，不是生成视频的人工标签或实验观测。使用既有球体渲染器，独立于原训练/验证/测试种子，以seed=40001生成256个96×96 RGB场景与二值掩膜。位置、颜色、模糊、噪声、影子、细红线由原渲染器随机产生。

将每个源场景缩放到24、48、96像素方形贴片，居中置入192×112白色画幅；图像用INTER_AREA，标签用nearest-neighbor。标签精确遵循这个离散变换规则，但并非重新计算抗锯齿连续物理边界。每个场景在3个贴片尺寸、3个输入策略和2个检测器中配对复用。

- native：直接输入192×112，充分卷积结构无需改权重。
- warp96：直接缩放至96×96。
- letterbox96：保持纵横比缩为96×56，上下各20像素白边。
- 检测器：既有冻结TinyUNet与颜色阈值 R>80且R>1.4G且R>1.4B。学习阈值固定0.5，不重新训练或选择权重。

预测掩膜经逆变换nearest-neighbor恢复至192×112后评测。所有场景都有球体；IoU以目标和预测并集为分母。质心误差单位为原始画幅像素，空预测按画面对角线222.2791px惩罚；非空预测率用于描述漏检。本实验没有无目标负例，不能据此评价假阳性、特异度或真实视频准确率。

独立抽样单位是256个程序场景；同一场景的尺寸、策略、检测器是配对重复，不是4608个独立样本。不涉及生物学重复。共形成3×3×2×256=4608条场景指标。独立随机种子不消除与训练共享渲染器带来的域限制。

## 统计方法与主要发现

逐场景计算掩膜IoU、质心误差、非空输出，条件内求均值。按场景索引进行2000次bootstrap，95%百分位CI，seed=20261110；与native比较使用同场景配对差。这是探索性分析，没有单/双侧显著性检验，没有P值或多重检验校正；CI只描述此合成生成器下的场景变异，不能外推真实视频。全部场景均纳入，没有按性能排除样本。

中等贴片（48px）的TinyUNet平均IoU从native的0.7954降至warp96的0.2986、letterbox96的0.1606。warp96−native配对差为−0.4968，95%CI [−0.5259, −0.4683]。最小贴片（24px）在两种96×96输入中非空预测率均为0，而颜色阈值均为1；这是已知有目标的合成场景中的漏检，不能解释为成功排除伪影。

最大贴片native时学习模型IoU为0.9650；表明“合成测试表现好”与“小目标条件下有效”是不同问题。由于letterbox同时改变目标像素尺度、padding和形状，不能单独归因于“保持纵横比”本身；native与缩小输入的比较支持尺寸/重采样敏感性，但尚未量化真实视频中各失效因素的贡献。

以下表格由实际JSON结果自动生成：

# Synthetic stress results

| Scene patch (px) | Input | Method | Mean IoU [95% CI] | Mean centroid error (px) | Nonempty fraction |
|---:|---|---|---|---:|---:|
| 24 | native | learned | 0.1616 [0.1340, 0.1889] | 120.9299 | 0.4570 |
| 24 | native | color | 0.6654 [0.6470, 0.6826] | 1.2017 | 1.0000 |
| 24 | warp96 | learned | 0.0000 [0.0000, 0.0000] | 222.2791 | 0.0000 |
| 24 | warp96 | color | 0.6589 [0.6434, 0.6743] | 0.6455 | 1.0000 |
| 24 | letterbox96 | learned | 0.0000 [0.0000, 0.0000] | 222.2791 | 0.0000 |
| 24 | letterbox96 | color | 0.5927 [0.5757, 0.6084] | 0.8167 | 1.0000 |
| 48 | native | learned | 0.7954 [0.7728, 0.8171] | 0.3766 | 1.0000 |
| 48 | native | color | 0.8028 [0.7835, 0.8214] | 2.0642 | 1.0000 |
| 48 | warp96 | learned | 0.2986 [0.2629, 0.3349] | 80.9271 | 0.6367 |
| 48 | warp96 | color | 0.7601 [0.7428, 0.7763] | 1.5744 | 1.0000 |
| 48 | letterbox96 | learned | 0.1606 [0.1334, 0.1875] | 120.8957 | 0.4570 |
| 48 | letterbox96 | color | 0.7319 [0.7152, 0.7481] | 1.7395 | 1.0000 |
| 96 | native | learned | 0.9650 [0.9615, 0.9682] | 0.2242 | 1.0000 |
| 96 | native | color | 0.8932 [0.8745, 0.9106] | 3.1234 | 1.0000 |
| 96 | warp96 | learned | 0.8339 [0.8207, 0.8463] | 0.3079 | 1.0000 |
| 96 | warp96 | color | 0.8065 [0.7873, 0.8248] | 3.5659 | 1.0000 |
| 96 | letterbox96 | learned | 0.7963 [0.7744, 0.8176] | 0.3267 | 1.0000 |
| 96 | letterbox96 | color | 0.8079 [0.7884, 0.8262] | 3.5963 | 1.0000 |

## Paired IoU contrasts

| Scene patch | Method | Contrast | Mean difference [95% CI] |
|---:|---|---|---|
| 24 | learned | warp96 minus native | -0.1616 [-0.1889, -0.1340] |
| 24 | learned | letterbox96 minus native | -0.1616 [-0.1889, -0.1340] |
| 24 | color | warp96 minus native | -0.0066 [-0.0247, 0.0118] |
| 24 | color | letterbox96 minus native | -0.0728 [-0.0892, -0.0557] |
| 48 | learned | warp96 minus native | -0.4968 [-0.5259, -0.4683] |
| 48 | learned | letterbox96 minus native | -0.6348 [-0.6611, -0.6093] |
| 48 | color | warp96 minus native | -0.0427 [-0.0521, -0.0330] |
| 48 | color | letterbox96 minus native | -0.0708 [-0.0803, -0.0612] |
| 96 | learned | warp96 minus native | -0.1311 [-0.1427, -0.1197] |
| 96 | learned | letterbox96 minus native | -0.1687 [-0.1885, -0.1493] |
| 96 | color | warp96 minus native | -0.0867 [-0.0927, -0.0800] |
| 96 | color | letterbox96 minus native | -0.0853 [-0.0912, -0.0788] |

## 产物与复现

源代码：本目录 `run_stress.py`、`summarize.py`；核心修复为仓库 `benchmark/physcheck/judge/frames.py` 和 `tests/test_frame_fidelity.py`。

`results/` 保存逐场景 `per_scene.json`、汇总 `summary.json`、配对对比 `paired_contrasts.json`、输入/真值 `scenes/`、全部预测 `predictions/`、版本哈希参数 `provenance.json`、图表 `scale_stress.png/.pdf/.svg` 及 `TABLES.md`。图展示全部场景点，误差线为按场景bootstrap的95%CI；PDF/SVG文字保留矢量，密集散点栅格化。

在当前仓库运行，输出目录必须尚不存在：

```bash
PYTHONDONTWRITEBYTECODE=1 /home/room115/pytorch_env/bin/python \
  research/stress_20261109T144046Z/run_stress.py \
  --checkpoint /home/room115/jhy/world-model-physical-consistency/.worktrees/video-bench/benchmark/results/training_pilot/best_model.pt \
  --output research/stress_20261109T144046Z/replay_NEW

/data/AI_Research/environments/python/sci/bin/python \
  research/stress_20261109T144046Z/summarize.py \
  --results research/stress_20261109T144046Z/replay_NEW

PYTHONDONTWRITEBYTECODE=1 /home/room115/pytorch_env/bin/python -m pytest -q -p no:cacheprovider
```

实际使用Python 3.12.3、torch运行时2.9.0+cu128（包元数据2.9.0）、numpy2.5.3、opencv-python-headless4.13.0.92；CPU、8线程。科研环境缺少torch，因此复用已有pytorch_env；绘图使用科研环境matplotlib3.11.2。检查点和训练代码SHA256在运行前与既有训练记录核对；源数据张量、当前脚本和软件版本均记录于provenance。没有安装新依赖。

## 局限与下一研究门槛

本轮证明冻结模型在所测合成尺度变化下失效，并完成抽帧输入修复。尚未完成真实生成视频定位准确率验证，未新增有效物理参数估计，也未完成Part B盲评。提示中的物体直径及视频FPS仍只是物理尺度/时间约定，跟踪改善本身不解决SI单位重力可辨识性。

下一步应将本次stress集视为开发诊断集，先研究多尺度合成训练或原生分辨率检测，再用预先冻结的新种子集和人工生成视频标签作独立评测；不能用本轮结果反复调参后继续称其为盲测。120帧待标注包已准备，人工标签不得由算法预测填充。考虑加入无球但有红线/红色干扰的负例，评测漏检与假阳性，之后再评价完整时序跟踪和尺度标定。

Part B还须人工核实控制视频是否实际出现预期现象，不能把控制prompt文字当真值；裁判需要正负例和混淆类别校准。当前框架允许任意`other`类别通过预期类别校准，可能使门槛偏松；本轮识别并记录此问题，未将其作为已修复或已校准结果。

## 最终复核

独立新目录重跑完成；per_scene、summary、paired_contrasts、provenance四份JSON逐字节一致。768幅合成场景、768份标签、4608份预测文件数量检查通过；三份检出16份退役文档的归档SHA256核对通过；原24视频审计JSONL和冻结检查点哈希保持一致。

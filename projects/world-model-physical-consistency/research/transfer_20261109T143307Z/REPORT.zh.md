# 视频世界模型物理一致性：进度核查与检测器迁移预检

日期：2026-11-09。研究问题：现有视频物理评测推进到哪里，合成训练的 TinyUNet 是否已有向生成视频迁移的证据？

## 进度与材料定位

远端私有仓库为 `gingersea/world-model-physical-consistency`。main 最新提交为 `08973c0f441051dde8e3dbd89cb10dc7a276aa9f`（2026-11-07），只增加三页课程 milestone；根目录 README/proposal/experiment-report 仍为旧气象方向，不能当作当前执行状态。

远端 `feat/video-bench` 停在 `2f8e8d2`；本机工作树 `/home/room115/jhy/world-model-physical-consistency/.worktrees/video-bench` 位于 `8f2abb7c66e2157eecd77d7df7a174fbb47a9194`，另有未提交的生成修复、训练脚本和实验产物。重要研究证据目前尚未完整同步至远端。本轮保留已有文件，新增输出独立保存；未提交或推送。

本次检索 `shared/research_plans/` 未发现已有本项目方案；已新增接续记录 `shared/research_plans/world_model_video_continuation_20261109T143307Z.md`。方向依据是最新 milestone、本机视频设计和 11 月 6 日进度报告。旧气象设计与视频设计有冲突，按最新执行证据接续视频，不能把两套实验混计进度。

11 月 6 日交接以“12 个工作包中前 6 个已验收”定义工程半程；这不代表论文完成度。本轮确认已有两模型推理、Part A 视频、合成测量验证和 TinyUNet 训练。Part B 只有协议/提示/抽帧骨架，未发现完成的校准盲评。11 月 6 日报告另记载拼图 RGB/BGR 问题待修，本轮未修复或验收该项。

## 数据来源与实验单位

冻结 Part A：Wan2.1 1.3B、CogVideoX-5B，各 4 个提示（落体、抛体、摆、水平运动）× 3 个种子（0/1/2），共 24 段视频。Wan 每段 81 帧、16 fps，CogVideoX 每段 49 帧、8 fps，共 1560 帧。本轮逐个重新解码、核对视频 SHA256、sidecar 帧数/FPS和原始审计记录。

单个视频是本轮描述性汇总单位，帧是相关重复测量；没有生物学重复。四个固定提示的选择限制外推范围，生成种子不构成人群抽样。TinyUNet 只有一个训练种子。本轮没有重新生成视频或重新训练模型。

## 处理步骤与分析参数

1. 核对既有训练脚本和冻结权重 SHA256；重新生成既有 256 张合成测试图及标签，核对测试集张量哈希。
2. 在 CPU 上重新评测既有模型；合成图的 IoU/质心误差使用原始训练脚本的评价函数。
3. 全部生成视频逐帧直接 INTER_AREA 缩放至 96×96；学习检测器固定 sigmoid >0.5。颜色基线为 R>80、R>1.4G、R>1.4B；两者使用同一缩放输入。这里没有连通域面积过滤和时序关联，不能与原有完整跟踪器的通过率混用。
4. 逐帧记录掩膜非空、两方法之间 IoU和质心差。两者都为空时方法 IoU记为缺失；单方为空时记为 0。按视频求均值，再按模型对视频等权汇总；方法 IoU只表示一致程度。
5. 以固定种子 20261109 随机排列视频匿名 ID，每视频均匀抽取 5 帧（含首尾），共 120 张全分辨率 RGB 帧。标注模板留空，身份映射单独保存。没有自动填造人工标签。

本轮是探索性、描述性诊断，没有随机分配干预或盲法准确率评价；匿名帧仅为后续人工验证准备。未计算 P 值或置信区间，也未开展多重检验。没有标签，无法计算生成视频分割准确率。固定小提示集与帧内相关性使按 1560 个独立样本推断不合理。所有失败视频均保留。

## 主要结果

冻结审计由当前输入 JSONL重新汇总：

| 模型 | 视频 | 原完整跟踪通过 | 原完整物理拟合通过 |
|---|---:|---:|---:|
| CogVideoX-5B | 12 | 5 | 0 |
| Wan2.1 1.3B | 12 | 11 | 0 |

总计 8 段跟踪失败、9 段标定拒绝、7 段拟合质量拒绝；这些是已有审计结果，本轮未用学习检测器改写原审计。0/24 表示无通过完整协议的可靠参数估计，不能当成物理正确率为零。

冻结检查点的合成测试复测（256 张既有图）：

| 检测器 | mean IoU | mean centroid error（96×96像素） |
|---|---:|---:|
| TinyUNet | 0.97037349 | 0.10946890 |
| 颜色阈值 | 0.90147135 | 2.76979205 |

原 GPU 报告的 TinyUNet IoU为 0.97041171；本轮 CPU复测差约 −0.00003822，权重、代码和测试张量哈希一致，跨设备浮点执行是可能解释，未进一步定位。不能声称逐位一致，也不能把重评同一测试集称作独立复制实验。

本轮全视频未标注预检：

| 视频来源 | 视频/帧数 | TinyUNet非空输出率 | 颜色基线非空输出率 | 两方法mean IoU |
|---|---:|---:|---:|---:|
| CogVideoX-5B | 12 / 588 | 31.63% | 43.88% | 0.51969 |
| Wan2.1 1.3B | 12 / 972 | 76.65% | 78.19% | 0.74021 |

非空输出率是每视频非空帧比例的平均。学习方法相对颜色基线分别少 12.24、1.54 个百分点。这个差异既可能反映去除假阳性，也可能是漏检，必须有人工标签才能区分。合成优势不足以支持在生成视频中替换完整跟踪器。逐提示配对图提示 CogVideoX 的 fall seed 0、proj seed 2 存在较大覆盖差异，可在完成统一标注后检查；不要因观察到分歧就只挑选这些帧报告性能。

## 图表、源代码与复现

当前目录包含 `screen_transfer.py`、`plot_screening.py`、本报告；`results/` 包含 `frames.json`、`clips.json`、`summary.json`、`synthetic_recheck.json`、`provenance.json`，以及 PNG/PDF/SVG 的 `transfer_screening` 图。图显示全部 24 个视频的成对输出非空率，不含误差线，不表示准确率。`annotation_frames/` 为120张匿名原图，`annotation_template.csv` 为空白人工标注模板。`predicted_masks/` 为抽样帧两方法的96×96预测，不能当真值。`restricted_blinding_key.json` 不提供给标注者。

从 `/data/AI_Research` 运行，`--output` 必须指向尚不存在的新目录：

```bash
PYTHONDONTWRITEBYTECODE=1 /home/room115/pytorch_env/bin/python \
  projects/world-model-physical-consistency/research/transfer_20261109T143307Z/screen_transfer.py \
  --source /home/room115/jhy/world-model-physical-consistency/.worktrees/video-bench \
  --output /data/AI_Research/projects/world-model-physical-consistency/research/transfer_20261109T143307Z/replay_NEW

/data/AI_Research/environments/python/sci/bin/python \
  projects/world-model-physical-consistency/research/transfer_20261109T143307Z/plot_screening.py \
  --results /data/AI_Research/projects/world-model-physical-consistency/research/transfer_20261109T143307Z/replay_NEW
```

依赖源工作树中的既有 `training_pilot.py`、检查点、视频和JSONL；实际路径及哈希均见 `provenance.json`。迁移诊断用 Python 3.12.3、PyTorch运行时 2.9.0+cu128（包元数据2.9.0）、NumPy 2.5.3、OpenCV 4.13.0、PyAV 17.0.1，CPU执行、8线程。绘图用科研环境 matplotlib 3.11.2。选择既有 pytorch_env是因为默认 sci环境缺少torch、cv2、imageio和pytest。

原项目回归测试实际运行：`PYTHONDONTWRITEBYTECODE=1 /home/room115/pytorch_env/bin/python -m pytest -q -p no:cacheprovider`，结果 **42 passed in 8.17s**。本轮脚本还实际执行了检查点/代码/测试集/视频哈希、样本唯一性及元数据一致性断言。

## 局限与下一步

- 缩放到96×96会改变纵横比、压缩小目标，合成训练图中的目标尺度与生成视频不同。结果只适用于这个明确的预处理。未来可比较保持纵横比的输入策略，但需先冻结独立标签和评价协议。
- “提示要求直径0.1 m”不等于图像中有可独立验证的真实长度；仅由像素轨迹和nominal FPS无法独立识别SI单位重力。即使跟踪改善，g仍是依赖尺度约定的条件估计；需明确镜头运动、透视、尺寸变化和时间重映射限制。
- 本轮不提供新物理一致性排名或新生成视频准确率；也未运行Part B裁判。旧方案的“首个”“无人研究”等新颖性表述未在本轮重新检索，不作为已核实结论。
- 下一科学门槛：人工标注120帧中的目标存在状态、原尺寸中心和/或二值掩膜，盲于模型及预测；对歧义帧预先约定处理并保留分母。若无唯一目标，区分生成/提示遵循失败与检测失败。人工标签只用于评测；若据此调模型，该集转为开发集，需按视频另留独立测试集，禁止帧级随机拆分泄漏。
- 然后比较完整颜色跟踪与学习检测的定位/时间连续性、尺度可靠性，再决定是否用新版本重跑Part A。不得放宽原门槛制造成功拟合。Part B须先修复并验证颜色保真、控制集校准、冻结裁判版本；校准样例的“正确答案”须依据实际画面而非提示文字。
- 代码、权重和报告的未提交版本尚未在远端统一；建议后续用独立分支整理可复现快照，再更新根目录入口指向视频工作。未公开发布任何研究材料。

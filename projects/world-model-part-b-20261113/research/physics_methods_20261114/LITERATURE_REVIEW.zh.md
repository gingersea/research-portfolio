# 物理一致性论文：方法调查与本项目取舍

检索日：2026-11-14。范围是直接关联本项目的8篇方法论文，不是系统综述。
阅读下列明确版本的方法、实验设置及相关官方实现；失败的PDF下载不计入阅读证据。
论文结论、代码行为、本次实验和下一步建议分开记录。

## 论文具体怎么做

| 工作及阅读版本 | 输入与评估对象 | 核心方法 | 对本项目的用法与限制 |
|---|---|---|---|
| [VideoPhy，2024，v2](https://arxiv.org/html/2406.03520v2)，§2–4、§6 | 688条固体/流体交互提示；人工评价生成视频 | SA/PC二值标签；测试每段3人、多数票；用人工标签微调VideoCon-Physics | 分别报告语义、物理及联合成功；原版标注同时给提示，不是完全去提示的PC协议 |
| [VideoPhy2，2025，v1](https://arxiv.org/html/2503.06800v1)，§2–4 | 动作场景；SA和PC分别标注，PC不看生成提示 | 1–5分；联合成功为SA≥4且PC≥4。候选规则从视频描述生成，另设无法确定及人工补充规则 | 本次复算公开人工测试表。不能用提示里的预期结果替代画面证据；不能把无法确定当通过 |
| [PhyGenBench / PhyGenEval，2024，v1](https://arxiv.org/html/2410.05363v1)，§3–5 | 160提示、27现象；语义与物理分开 | 检索关键现象帧及邻近5帧，检查前后事件顺序，再评价整段自然性；分数离散后聚合 | 借鉴事件定位和时序证据；未运行其GPT-4o/VQAScore/视频模型集成，普通拼图不等于该协议 |
| [Physics-IQ，2025，v1](https://arxiv.org/html/2501.09038v1)，数据及Metrics章节 | 真实实验初始帧/3秒视频，预测后5秒；双次实拍估计自然变动 | 运动激活图的空间、时空、加权IoU与MSE，以实拍间变动归一化 | 需要相同初态及参考视频。旧自由T2V片段缺少这些条件，不能直接算其物理分数 |
| [Physics-IQ Verified，2026，v1](https://arxiv.org/html/2606.18943v1)，§3、附录C | 修订提示、真实参考和激活图 | 明确初态/触发动作；去装置和偶发伪激活；每样本先归一化、截断、等权聚合 | 报告测量覆盖率和分母；背景运动、相机变化不应成为物理信号。未运行完整基准 |
| [Principia，2026，v1](https://arxiv.org/html/2609.04200v1)，§3–4、附录A | 真实首帧条件下的双物体、8类牛顿现象 | 同场景比值/相等关系免绝对尺度标定；SAM3轨迹；方向筛选后算关系分数 | 是替代旧SI拟合的优先候选，但须控制同步释放、几何、材质；任意双球视频不是受控实验 |
| [VPT，2026，v1](https://arxiv.org/html/2607.04653v1)，§3–4 | Wan生成；训练时附加光流及角色图 | 视频/光流/角色三流建模；独立噪声时间与辅助损失退火；另有早期检查点自引导 | 本次选发布权重实际推理；不重训，不声称复现缺少的早期检查点自引导 |
| [LaMo，2026，v1](https://arxiv.org/html/2605.23878v1)，§3、§4.1 | 视频VAE潜变量；CogVideoX骨干 | 时间差潜变量空间均值作训练漂移目标；小CNN学运动场，低噪声时对噪声预测梯度引导 | 可研究的训练方向；宏观差分统计不是完整LaMo，平滑正则也不等于复现 |

## 对原方向最有用的三个改变

第一，重新定义测量目标。单物体像素抛物线没有尺度、透视和相机控制，不能可靠推断SI参数。
Principia用同场景关系减少标定需求，但没有免除受控初态、可靠轨迹和适用条件检查。
例如重力比较高度比与落地时间平方比；摆动比较长度比与周期平方比。
还应同时报告未满足运动/跟踪门的比例，防止仅对成功子集给漂亮均值。
这是我们对适用范围的判断。[方法来源](https://arxiv.org/html/2609.04200v1#S4.SS1)

第二，裁判先看可观察证据，再谈物理。VideoPhy2把PC从生成提示中隔离；候选规则允许无法判断。
PhyGenEval另查事件先后，提示应保存事件附近原生帧，而非仅固定8帧。
两套尺度和聚合不同，不能拼成未经验证的统一分数。
[VideoPhy2协议](https://arxiv.org/html/2503.06800v1#S3)、
[PhyGenEval协议](https://arxiv.org/html/2410.05363v1#S4)。

第三，生成方法和评估方法要分别验证。VPT有可用的小型适配权重，适合现有Wan缓存；
LaMo还需经过训练的运动预测器。两者改善学习到的运动先验，不保证守恒定律。
后续即使自动裁判给高分，也须先用确定标签正反例和人工子集校准。
本次选择VPT是资源与可复现性取舍，不是认定它最好。
[VPT方法](https://arxiv.org/html/2607.04653v1#S3)、
[LaMo方法](https://arxiv.org/html/2605.23878v1#S3)。

## 代码与资产核查

- VPT：[代码固定版本](https://github.com/Tom-zgt/VPT/tree/3097bcd0267203a8192567d6eb2a9831b8015945)，
  [权重固定版本](https://huggingface.co/zhengzhou/VPT/tree/edacaa9709af842a017361d4140fffac3b74fecc)。
  `custome.py`有特殊三流采样，普通Wan LoRA加载不完整。
  README基准配置inner CFG=2.2，辅助引导=0；代码等效普通文本CFG=3.2。
  本次专门加入3.2基线；这不是论文的跨检查点自引导。
- VideoPhy2：[代码固定版本](https://github.com/Hritikbansal/videophy/tree/01d366c979e4a82875852adbf427ccc036e7373a)，
  [公开人工表固定版本](https://huggingface.co/datasets/videophysics/videophy2_test/tree/90b81ffa54f565d9e40e83b7a2c247ba1dccfa2b)。
  自动评估器为mPLUG-Owl视频/LLaMA7B，官方采32个分段中心帧。
  大权重未下载完成，本次没运行自动裁判；人工表复算不是裁判校准。
- Principia：[代码固定版本](https://github.com/PrincipiaBench/Principia/tree/893e003804f0994ea6467b5a756ab657b82d4df9)。
  README明确SAM3权重需授权；本次核查数值打分和筛选，没跑SAM3全流程。
- LaMo：[作者项目页](https://lamo-ai.github.io/)、
  [官方仓库资产说明](https://github.com/Applied-Intuition-Open-Source/LaMo#installation)。
  README明确权重未包含，推理需要LoRA和predictor组合。本次未取得可验证的完整组合，未执行。

源码清单、数据revision和权重SHA256见本目录manifest及外部缓存`source/*/SOURCE_MANIFEST.json`。
各论文分数不直接排榜，因为输入、提示、评估器和阈值不一致。

# 文献与检索记录

日期：2026-11-19。该检索为面向面试提案的定向探索，非系统综述。根据论文原始页面、摘要及作者仓库整理，未独立复现论文结果。下表注明各项论述的依据与边界。

检索式：

- `LLM agents dynamic goal updating replanning feedback goal revision benchmark paper`
- `Reflexion Self-Refine ReAct plan and solve agent replanning paper`
- `AgentChangeBench goal shifts benchmark`
- `Benchmarking the responsiveness of LLM-based agentic systems under controlled mid-task perturbations`

检索及核查范围为 arXiv 原文/摘要、ICLR 会议页、作者仓库；不使用未经核验的排行榜数字。

| 来源 | 直接支持的有限论述及使用边界 |
|---|---|
| Yao et al., ReAct, ICLR 2023；[arXiv v3](https://arxiv.org/abs/2210.03629v3) | 推理、动作和环境观察交替进行。本文 R 组是 ReAct 风格设计，不声称严格复现实验。 |
| Shinn et al., Reflexion, 2023；[arXiv v4](https://arxiv.org/abs/2303.11366v4) | 语言反馈与跨试验 episodic memory，不更新权重。提案的一次回合内反思并非原方法的完整复现。 |
| Madaan et al., Self-Refine, 2023；[arXiv v2](https://arxiv.org/abs/2303.17651v2) | 同一模型生成、反馈和修订，无额外训练。不能据此认定动态 Agent 子目标更新有效。 |
| Huang et al., ICLR 2024；[会议原始页面](https://proceedings.iclr.cc/paper_files/paper/2024/hash/8b4add8b0aa8749d80a34ca5d941c355-Abstract-Conference.html) | 所研究推理任务中，无外部反馈的自我纠错可能失败或降低表现；不能推广成所有反思必然无效。 |
| Liu et al., AdaPlanBench, COLM 2026；[原文 v2](https://arxiv.org/html/2606.05622v2) | 307 个家庭任务，逐步揭示约束；第 4 节及 C.6–C.7 支持区分约束有效性与最终成功，反馈也可能损伤既有约束。 |
| Zhang et al., DynaRev, 2026 年预印本；[原文 v1](https://arxiv.org/html/2610.40133v1) | 181 个网页任务，在语义节点引入用户修订；即时响应与最终履约不同。模型裁判有误差，且与固定最终目标的本方案不同。 |
| Balyan & Sharma, Scientific Reports, 2026；DOI `10.1038/s41598-026-72720-7`；[作者仓库](https://github.com/AyushBalyan/Benchmark) | README 已比较 ReAct、Reflexion、Plan-and-Execute，以及目标、工具和约束扰动、漂移与恢复。架构比较加扰动不是本提案首创。 |
| Rana et al., AgentChangeBench, NeurIPS 2025 workshop；[arXiv v1](https://arxiv.org/abs/2510.18170v1) | 补充阅读：恢复时间和冗余工具调用指标。属于 workshop，不能写成 NeurIPS 主会论文。 |

重要限制：Scientific Reports 出版商完整页面未能通过浏览工具访问。重叠方法判断仅依据已核查的作者仓库 README；不声称核查了出版全文、实验细节或全部结果。

此前阅读的 Curse of CoT、LogiDynamics、MemLens、NewtonBench 仅作为选题启发；正文不依赖这些论文的具体数值或机制结论。

## 方案状态

本次交付是文字研究方案，尚未实现模拟器、调用模型或运行实验。十个开发题、二十个测试基础题、360 次运行、预算及四周计划全部为拟议参数。已核对六组设计及 `20 × 3 × 6 = 360`，但这不代表统计功效或实验可行性已获实测验证。

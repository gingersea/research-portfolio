# 物理一致性论文调查及发布方法试用

本轮按用户要求回到物理一致性相关论文，调查8篇方法，并实际试用VPT发布权重。
这是旧实验收尾后的独立跟进；旧Part B校准失败结论仍成立。

- [调查与实验报告](../research/physics_methods_20261114/REPORT.zh.md)
- [8篇论文的方法、输入、评价及限制](../research/physics_methods_20261114/LITERATURE_REVIEW.zh.md)
- [运行前固定试验协议](../research/physics_methods_20261114/TRIAL_PROTOCOL.md)
- [实际6段视频验收](../research/physics_methods_20261114/generation_audit.json)
- [碰撞场景三臂对照](../research/physics_methods_20261114/inspection/collision_comparison.jpg)
- [复现及交付说明](../research/physics_methods_20261114/HANDOFF.md)

已复算VideoPhy2公开人工表3,397行，并在382个共同提示上另报平衡描述表。
已完成2提示×3臂、固定seed42的6段生成，共486帧，校验解码、形状、帧率、噪声与哈希。
VPT在匹配对照的一个碰撞样例中保留了末段蓝球，但接触附近仍有形变；这只是非盲观察，
没有新人工标签或校准裁判，不构成物理提升率。

下一候选方向是受控初态下的双物体关系测量及事件证据校准，避免继续以自由T2V像素轨迹直接推断SI参数。

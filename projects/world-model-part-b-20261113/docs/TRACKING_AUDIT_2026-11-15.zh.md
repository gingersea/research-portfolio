# 双物体跟踪失败诊断与三维方法阅读

逐帧复现旧388帧检测结果后，确认：24帧角标拒绝来自额外绿色伪影；固定颜色和全图唯一性也会丢掉仍可观察的前景。

新的空间角标匹配消除该误判；颜色无关原型使B/42的可用帧从8/97增至58/97，但四段仍均不通过覆盖门。新增身份替换挑战出现4/4错误身份声称，所以原型没有升级为物理测量器，原扩样决定不变。

见 [完整诊断与修正报告](../research/tracking_audit_20261115/REPORT.zh.md) 和 [CoTracker3、SpatialTracker、MegaSaM、TAPIP3D方法精读](../research/tracking_audit_20261115/READING_2D_TO_3D.zh.md)。

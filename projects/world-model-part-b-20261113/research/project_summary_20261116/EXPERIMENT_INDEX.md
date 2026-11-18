# 历次研究与证据索引

当前整体状态与路线见[项目总结](../../docs/PROJECT_SUMMARY_2026-11-16.zh.md)。
这里列出18个既有研究目录；表内状态只表示已执行证据的范围，不是新增实验或通用有效性认证。
早期Part A和检测器试点另存于benchmark/results与report，不计入这18目录。

|研究目录|工作|状态/范围|原报告|
|---|---|---|---|
|transfer_20261109T143307Z|Transfer screening|complete_unlabeled|[报告](../transfer_20261109T143307Z/REPORT.zh.md)|
|stress_20261109T144046Z|Scale stress and sampling fixes|complete_scoped_controls|[报告](../stress_20261109T144046Z/REPORT.zh.md)|
|multiscale_20261109T191120Z|Matched multiscale training|primary_endpoint_not_improved|[报告](../multiscale_20261109T191120Z/REPORT.zh.md)|
|part_b_20261113T114040Z|Part B production and isolated judgments|production_complete_calibration_invalid|[报告](../part_b_20261113T114040Z/REPORT.md)|
|closeout_20261114|Sparse-frame information limit|complete_scoped_counterexamples|[报告](../closeout_20261114/results/REPORT.md)|
|action_world_stage0_20261114|Action-conditioned linear-state pilot|complete_toy_side_study|[报告](../action_world_stage0_20261114/results/REPORT.md)|
|physics_methods_20261114|Literature, public labels and VPT trial|complete_scoped_trial|[报告](../physics_methods_20261114/REPORT.zh.md)|
|principia_gravity_20261114|Controlled two-object gravity relation|scoped_calibration_passed|[报告](../principia_gravity_20261114/REPORT.zh.md)|
|principia_i2v_20261115|Native 2-state x 2-seed I2V|generation_complete_measurement_gate_failed|[报告](../principia_i2v_20261115/REPORT.zh.md)|
|tracking_audit_20261115|Coverage audit and identity counterexample|identity_challenge_failed|[报告](../tracking_audit_20261115/REPORT.zh.md)|
|tracking_validation_20261115|Native 2D tracking and geometry ablation|baseline_complete_production_gate_failed|[报告](../tracking_validation_20261115/REPORT.zh.md)|
|observation_gate_20261115|Independent RGB gate and native TAPIP3D|scoped_gate_complete_recovery_gap|[报告](../observation_gate_20261115/REPORT.zh.md)|
|identity_recovery_20261115|Recovery V1, I2V reference and depth chain|mixed_scoped_results_real_gap|[报告](../identity_recovery_20261115/REPORT.zh.md)|
|identity_recovery_multiscale_20261115|Recovery V2 on new controls|scoped_positive_result_not_real_validation|[报告](../identity_recovery_multiscale_20261115/REPORT.zh.md)|
|paper_benchmark_20261115|DAVIS and calibrated PStudio pilot|external_label_baselines_complete|[报告](../paper_benchmark_20261115/REPORT.zh.md)|
|measurement_validation_20261115|Acceptance calibration and error attribution|precision_coverage_tradeoff_not_general_fix|[报告](../measurement_validation_20261115/REPORT.zh.md)|
|integrated_framework_20261115|Unified entry and fresh I2V chain|engineering_complete_scientific_gaps|[报告](../integrated_framework_20261115/REPORT.zh.md)|
|integrated_matrix_20261115|325-input matrix and diagnostic opinions|engineering_equivalence_complete|[报告](../integrated_matrix_20261115/REPORT.zh.md)|

状态中的failed/invalid均保留为实测失败，scoped不代表已完成自然域验证。
已查看的DAVIS/控制不能作为调参后的全新留出；编码、点、帧和重复运行不作为新增独立样本。
目录内handoff_receipt.json（若有）指向版本化共享交接。source_snapshots保留对应阶段代码。
完整机器来源和SHA256见[evidence_catalog.json](evidence_catalog.json)。

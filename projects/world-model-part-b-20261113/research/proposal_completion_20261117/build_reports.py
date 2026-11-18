"""Render the proposal report from immutable original and new analysis outputs."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
sys.path.insert(0, str(PROJECT / 'benchmark'))
from physcheck.report.render import render_part_a, render_part_b, render_prompt_fits, render_categories


def load(path):
    return json.loads(path.read_text())


def documents():
    a = load(PROJECT / 'benchmark/results/raw/part_a_summary.json')
    a_rows = [json.loads(s) for s in (PROJECT / 'benchmark/results/raw/part_a.jsonl').read_text().splitlines()]
    b = load(PROJECT / 'research/part_b_20261113T114040Z/run/results/leaderboard/part_b_leaderboard.json')
    detector = load(PROJECT / 'benchmark/results/raw/detector_validation.json')
    d = load(ROOT / 'analysis/davis_replication.json')
    geom = load(ROOT / 'analysis/full3d.json')
    pc = load(ROOT / 'analysis/human_pc_probe.json')
    gravity = load(ROOT / 'analysis/gravity_judge_gate.json')
    tests = load(ROOT / 'tests_final.json')
    dm = d['summary']['by_domain']['davis']['video_mean_metrics']['co']
    gm = geom['summary']['by_domain']['pstudio']['video_mean_metrics']
    geometry = ['| Video (150 frames; 8 queries) | Co+depth full EPE (cm) | TAPIP3D full EPE (cm) | Co invalid / GT-visible point-times | Co / TAP common-subset EPE (cm) |',
                '|---|---:|---:|---:|---:|']
    for row in geom['cases']:
        co, tap = row['measurements']['co'], row['measurements']['tap']
        epe = 'N/A' if co['raw_epe3d_m'] is None else f'{co["raw_epe3d_m"] * 100:.2f}'
        geometry.append(f'| {row["case"]} | {epe} | {tap["raw_epe3d_m"] * 100:.2f} | {co["invalid_3d_point_times"]}/{co["visible_point_times"]} | {row["co_common_epe_m"] * 100:.2f} / {row["tap_common_epe_m"] * 100:.2f} |')
    geometry = '\n'.join(geometry)
    detector_rows = '\n'.join(f'| {r["kind"]} | {r["param"]} | {r["true"]:.5f} | {r["estimated"]:.5f} | {r["rel_err"]:.6f} |' for r in detector['results'])
    n_a = len(a_rows)
    n_b = len(b['diagnostic_only']['per_video'])
    all_invalid = gm['co']['count_totals']['invalid_3d_point_times']
    all_visible = gm['co']['count_totals']['visible_point_times']
    english = f'''# Video physical consistency: measurement validity before generator ranking

Research report, 2026-11-17. This is an experimental closeout and paper draft, not a validated public leaderboard.

## 1. Introduction

Visual plausibility alone does not identify physical correctness. We implemented the October 16 proposal's two-part local evaluation and then tested the measurement and judging prerequisites exposed by its failures. The scientific question is whether generated videos support defensible physical estimates and calibrated judgments, with explicit coverage and abstention. The current results establish an executable framework and several failure boundaries; they do not establish a ranking of the generators.

## 2. Related work

The source papers below were rechecked on their primary arXiv pages. Existing benchmarks include open and commercial models, so the original proposal's blanket claim that this research fills an otherwise absent open-model niche is not retained.

| Study | Scope and metric approach | Model-access coverage and relation to this project |
|---|---|---|
| [Physics-IQ](https://arxiv.org/abs/2501.09038) | Physical-principle continuation tests with controlled reference outcomes | Includes open and commercial generators; motivates reference-based evaluation |
| [VideoPhy2](https://arxiv.org/abs/2503.06800) | Action-centric physical consistency; human PC labels and automated evaluation | Open and commercial models; public labels support our independent probe |
| [PhyGenBench](https://arxiv.org/abs/2410.05363) | Physical commonsense scenarios and structured multimodal evaluation | Evaluates multiple open and commercial systems; motivates task-specific evidence |
| [T2VPhysBench](https://arxiv.org/abs/2505.00337) | Physical-law-oriented text-to-video evaluation | Includes open and commercial models; not an exclusively closed-model benchmark |
| [Principia](https://arxiv.org/abs/2609.04200) | Controlled relational physical tests across eight phenomena | Relational measurement motivates our restricted two-object gravity test; we do not reproduce its complete benchmark |
| [Physical Law Perspective](https://arxiv.org/abs/2411.02385) | Quantitative tests of physical laws in generated videos | Supports separating numerical measurement from visual realism |

For measurement, [CoTracker3](https://arxiv.org/abs/2410.11831) provides point tracks, [TAPIP3D](https://arxiv.org/abs/2504.14717) provides native 3D point tracking, and [TAP-Vid](https://arxiv.org/abs/2211.03726) defines public tracking evaluation. Our official-metric DAVIS evaluation is distinct from generator physical accuracy. Depth Anything V2 and the authors' released tracking models are used as fixed inference baselines, not trained improvements.

## 3. Benchmark design and methods

Part A uses local Wan2.1-1.3B and CogVideoX-5B, four prompts per model and three seeds per prompt. Tracking precedes scale/time admissibility and fitting. Unavailable g/T estimates are missing, never zero error. Known-parameter procedural animations validate the scoped detector/fitter before generated-video evaluation. Those controls do not validate domain transfer.

Part B uses complex-scene prompts, uniformly sampled static frames, opaque identities, fixed JSON criteria and separate contexts. The original production judgments and calibration failure remain immutable. A prompt requesting a physical violation is not a verified positive label. Static frames cannot establish every intervening event. The former proposed Claude backend was not used for these runs: the original receipts document isolated GPT-family sessions; the new visual study uses a pinned local Qwen checkpoint. Generic paid API routes are not claimed to have been tested.

The November 17 protocol was frozen in [PROTOCOL.md](../research/proposal_completion_20261117/PROTOCOL.md) before scoring. We added fresh inference to the same CLI for all existing DAVIS videos and the same three PStudio examples, extending them from 64 to 150 frames with the same eight selected queries each. Each query runs independently; inference receives video, official initial queries and supplied intrinsics where applicable. Future truth is scored separately. CoTracker plus estimated depth and native TAPIP3D share depth/camera conventions. Invalid lifts are retained in the GT-visible, post-query denominator. Full EPE is unavailable if any required point is invalid; conditional finite/common-subset diagnostics are separate.

The new VideoPhy2 probe fixes 24 author-labeled source videos: 12 with human PC<=2 and 12 with PC>=4, hash-ordered within class. Four per class calibrate; eight per class evaluate. All planned source videos were obtained without replacement. Model names, captions and labels are withheld from judging. Qwen2.5-VL-7B-Instruct revision `cc594898137f460bfe9f0759e9844b3ce807cfb5` receives 32 evenly sampled source frames as two indexed montages and deterministic decoding. All calibration classifications must be correct. Evaluation continues diagnostically when this gate fails. This is neither the author's AutoEval reproduction nor a calibration of the old eight-frame Part B. The human label is PC only; SA and joint scores are not predicted here.

The same fixed Qwen model first judges 12 prior gravity controls, then the four existing native I2V videos in isolated contexts, with all-frame sheets and the previous applicability rubric. These are new AI opinions, not human labels. A separate three-color visual-channel probe checks that image input is functioning; it cannot validate temporal reasoning.

Video/scene is the experimental unit; frames, queries, encoding variants and reruns are dependent. DAVIS and the three PStudio clips were already exposed during prior development, so fresh inference is a reproduction/length extension, not unseen validation. The balanced human-PC test describes only its fixed sample. Its accuracy interval resamples caption clusters 10,000 times, seed 20261117. No p-values or population claims are appropriate for the three selected geometry examples. No training or optimization was done in this round.

## 4. Results

### 4.1 Part A and detector gate

{render_part_a(a)}

{render_prompt_fits(a, a_rows)}

All {n_a} generated clips lack a valid complete physical fit. The g/T estimates and relative errors are unavailable; this is not zero physical correctness. Prompt types without g/T do not acquire invented values. Per-clip failure reasons are in the frozen [raw records](../benchmark/results/raw/part_a.jsonl).

| Synthetic condition | Parameter | Known value | Estimate | Relative error |
|---|---|---:|---:|---:|
{detector_rows}

The synthetic detector/fitter gate is `{detector['passed']}`. These numeric recoveries support only the known procedural conditions. Transfer to generated small/deformed targets remains unvalidated.

### 4.2 Original Part B

{render_part_b(b)}

{render_categories(b)}

Production judgments cover {n_b} clips; the original gate is `{b['valid']}`. All {len(b['calibration']['unverifiable_controls'])} generated controls were unverifiable as intended positives under that protocol. The [original leaderboard artifact](../research/part_b_20261113T114040Z/run/results/leaderboard/part_b_leaderboard.json) retains diagnostic counts, including categories, but no valid rates or ranking are reported. A new judge's result cannot retroactively repair those inputs.

### 4.3 Fresh labeled measurement runs

DAVIS: {d['videos']} videos, {d['queries']} queries; AJ {dm['average_jaccard'] * 100:.2f}%, APD {dm['average_pts_within_thresh'] * 100:.2f}%, OA {dm['occlusion_accuracy'] * 100:.2f}%. Maximum coordinate difference from the frozen prior prediction is {d['max_xy_change']:.1f} pixels, with no visibility differences. This verifies adapter reproduction, not new generalization.

{geometry}

Across the three full-length PStudio examples, Co+depth misses {all_invalid}/{all_visible} required point-times. Its full macro EPE is N/A. Its finite-subset macro EPE of {gm['co']['finite_epe3d_m'] * 100:.2f} cm must not be compared directly with TAPIP3D's full macro EPE of {gm['tap']['raw_epe3d_m'] * 100:.2f} cm. The final column uses identical common-valid points per video for descriptive comparison, while retaining the missing coverage separately. Offline full-sequence inference can change earlier predictions too; length extension alone does not identify the cause of any difference from the previous prefix experiment.

### 4.4 Independent local visual judging

Gravity calibration: {gravity['correct']}/{gravity['calibration_cases']} correct; valid=`{gravity['valid']}`. Both unequal-gravity violations were missed, and eight required abstentions were called PASS. The four I2V first replies (three PASS, one FAIL) therefore remain uncalibrated diagnostics. Their old numeric physical relations remain unmeasurable. Image-channel sanity checks passed all three colors, which only excludes a completely disconnected image channel.

Human-PC calibration: {pc['calibration_correct']}/{pc['calibration_planned']} correct; valid=`{pc['protocol_valid']}`. Test: {pc['evaluation_correct']}/{pc['evaluation_planned']} correct, diagnostic accuracy {pc['diagnostic_accuracy'] * 100:.1f}%, caption-cluster bootstrap 95% interval [{pc['caption_cluster_bootstrap95ci'][0] * 100:.1f}%, {pc['caption_cluster_bootstrap95ci'][1] * 100:.1f}%], {pc['caption_clusters']} test caption clusters. All test clips were called consistent: TP={pc['tp']}, FP={pc['fp']}, FN={pc['fn_or_missing']}, TN={pc['tn']}; missing/failed={pc['unavailable_or_failed']}. This equals accepting every clip in the balanced test and supplies no evidence of useful physical discrimination. Here positive means physically consistent, so FP is false acceptance of a human-inconsistent clip, distinct from falsely accusing a consistent clip of violation. No valid generator ranking follows.

![Full-length geometry and failed judge gates](../research/proposal_completion_20261117/figures/summary.png)

## 5. Discussion and limitations

The executable chain separates generation success, track existence, physical admissibility, numerical accuracy and calibrated judgment. Their distinction explains why no runtime errors and valid synthetic controls can coexist with failed scientific endpoints. Reappearance/identity filtering can raise precision through rejection; it must be assessed with retained coverage. Known-camera point geometry does not validate unknown-camera object-center relations in generated videos. Neither surface-point accuracy nor an assumed camera recovers SI physical constants without further evidence.

The Qwen results characterize this checkpoint, montage representation and frozen prompt. They do not establish that every VLM or video-conditioned judge fails. The human-PC sample deliberately selects extreme labels, excludes PC=3, is small and balanced, and cannot estimate production prevalence. Temporal montage resolution may lose evidence even when all source frames are represented. Existing generator clips still lack independent human physical labels; public point labels and author-provided PC labels support their own endpoints only. Procedural ground truth, human annotations, geometry ground truth and AI opinions remain separate.

Historical detector scale and multiscale experiments, identity recovery and confidence thresholds are indexed in the [November 16 summary](../docs/PROJECT_SUMMARY_2026-11-16.zh.md); no historical rerun is counted as new independent data. This work has not demonstrated a new general world model, a robust physical judge or a state-of-the-art tracking method.

## 6. Future work and conclusion

The next scientific prerequisites are verified positives/negatives with the same observed evidence as the judge, external small-target/identity validation with coverage, defensible camera/depth calibration, and human labels for this project's own generated clips. A redesigned judge or measurer must freeze a new protocol and use new validation examples. Generator expansion and constrained video fine-tuning remain conditional on these prerequisites; tuning on the failed test set would invalidate its future use as confirmation.

The conclusion is an experimentally supported measurement-validity failure, with a runnable baseline framework and explicit failure accounting. The original numerical-physics and calibrated-ranking objectives remain unmet. Experimental execution and a complete report do not turn failed gates into successful acceptance.

## Reproducibility, availability and AI assistance

Run commands and caveats are in [REPRODUCE.md](../research/proposal_completion_20261117/REPRODUCE.md). The final recorded test output is `{tests['stdout'].strip().splitlines()[-1]}`; exit={tests['exit']}. LSP analysis was unavailable because the configured server is not installed. A local copied-source/cache replay verifies only that stated scope, not a clean-machine neural reproduction. All first replies, frozen manifests, failed attempts, model hashes and runtime receipts are retained. Qwen weights are Apache-2.0; source videos retain their original rights, and a dataset-card license does not establish rights to redistribute every linked video. No new public upload was made. The old gingersea target is prohibited by current workstation instructions; local handoff is the reviewable deliverable.

AI assistance: GPT-family assistance was used for implementation, analysis and writing; Qwen supplied the explicitly uncalibrated visual opinions. A separate local GLM-family review is recorded with its actual coverage and [rulings](../research/proposal_completion_20261117/review/RULINGS.md): its broad output was truncated/repetitive and its bounded follow-up made unsupported allegations, so neither is claimed as a comprehensive independent pass. Scientific claims are based on saved inputs and computed outputs rather than model agreement.
'''
    zh = f'''# Proposal 实验收尾与结果（2026-11-17）

## 结论

完整流程已经运行，新增公开数据统一新推理、完整长度三维对照、人工PC标签原视频盲评与英文主报告。科学门仍失败，不能声称 proposal 的数值物理与有效排行榜目标全部完成。新增生成按冻结协议停在门前，没有通过调参或换样本凑分。

## 本轮新增结果

- DAVIS {d['videos']}段、{d['queries']}查询重新运行原生CoTracker3：AJ {dm['average_jaccard']*100:.2f}%、APD {dm['average_pts_within_thresh']*100:.2f}%、OA {dm['occlusion_accuracy']*100:.2f}%；与旧预测逐坐标一致。这是已知数据复现。
- PStudio同3段、同8查询扩至每段150帧。Co+深度在{all_visible}个GT可见点时中缺{all_invalid}个，完整3D EPE保留N/A；TAPIP3D完整宏均值{gm['tap']['raw_epe3d_m']*100:.2f} cm。Co有限子集{gm['co']['finite_epe3d_m']*100:.2f} cm不可直接与TAP完整分母作胜负比较。
- 独立本地Qwen重力校准{gravity['correct']}/{gravity['calibration_cases']}，遗漏两处明确违反并把8个应弃权控制判PASS，协议无效。4旧I2V首次盲评已实际执行，但不能作为有效物理标签；旧数字关系仍不可测。
- VideoPhy2作者人工PC标签对应24原视频全部取得、无替换。先8校准、后16验证；校准{pc['calibration_correct']}/{pc['calibration_planned']}失败。测试{pc['evaluation_correct']}/{pc['evaluation_planned']}（{pc['diagnostic_accuracy']*100:.1f}%，描述性95%区间{pc['caption_cluster_bootstrap95ci']}），全部预测一致，8段人工不一致视频全部误接受。这只是诊断结果，不是有效排名。
- 原Part A {n_a}段仍无完整有效拟合，g/T及误差N/A；原Part B {n_b}生产段与8控制已生成/抽帧/裁判，原校准无效，排名N/A。
- 最终测试：`{tests['stdout'].strip().splitlines()[-1]}`；图表与数值由保存JSON渲染，首次输出和所有失败保留。

{geometry}

同有效子集比较只能描述该子集。3段PStudio不是3个新独立验证场景；150帧离线推理也可能改变前缀预测，不能把差异因果归结为后半段。DAVIS无物理标签，PStudio是点几何标签，VideoPhy2的PC是物理人工评分，均不能替代本项目生成片的人类标注。

## 对照原计划与路线图

|原任务|执行状态与证据|科学验收/残留|
|---|---|---|
|1 环境与目录|既有两模型本地环境/脚手架已实际运行；本轮版本见tests_final.json与网络收据|通过范围仅限记录环境，非干净机器安装|
|2 模型注册/生成/prompt|Wan与Cog既有生成矩阵、固定种子与哈希|完成本地小矩阵；不是所有模型|
|3 合成真值动画|已知g/T等检测验证与后续程序控制保留|局部验证通过，不代表真实域|
|4 跟踪|原跟踪、原生CoTracker与完整TAPIP3D、公开新推理已执行|可靠最小目标、真实身份/覆盖仍未解决|
|5 物理拟合与门|合成参数恢复通过；生产拟合逐段失败理由保留|真实生成片尺度/可测性未通过|
|6 Part A生产|24视频全部生成→跟踪→尝试拟合|0/24有效完整拟合；必须给出可靠物理数值的目标未满足|
|7 Part B抽帧与控制|原48+8已完成；本轮增加固定标签原视频与32帧输入|请求违反的prompt不等于已知正例|
|8 裁判协议与校准|schema、盲化、首次输出、校验与新本地裁判已运行|原门失败；新两个门同样失败；有效校准未完成|
|9 CLI|生成/分析既有入口；统一relations增加fresh-public、严格缓存与失败计分|端到端工程可运行不等于物理准确|
|10 Part B生产与聚合|原48生产+8控制完整，保留无效leaderboard；新24人工标签探针已跑|有效Part B物理排名仍N/A|
|11 英文报告/表格|report/report.md、模型×prompt、类别表、文献比较、方法/讨论/披露齐全|完整负结果底稿，不是成功性能论文|
|12 清理/README/交付|沿用已归档气象方向；本轮README、摘要、本地复制复现包|远端提交/推送未做：最新AGENTS禁止gingersea；不宣称远端交付完成|

11月16路线：P1公开统一新推理与固定协议已完成；P2冻结二维/二维+深度/三维及独立视觉比较已执行，校准有效性失败；P3新增生成依赖未满足，按协议停止扩样；P4已形成负结果综合稿，尚无通过外部端到端验证的新方法。训练/大规模优化未启动。

## 还要做什么

1. 另定新协议与新验证集，使用在同一观察模态下真实可识别的已知正负例验证裁判。当前失败样本只能作开发/诊断，不能继续调到通过后称留出验证。
2. 在外部小目标、遮挡重现、歧义标签上同时报告误接受与覆盖，再验证三维和相机链路；可靠最小目标与假阳性控制仍是开放科学任务。
3. 为项目自产生成片取得独立人类物理标注；现有公开标签已支持本轮指定终点，但没有填补自产片标签。
4. 上述门通过后才继续小规模生成矩阵或优化。公开提交另需可用目标与发布确认；旧gingersea目标受当前工作站规则明确禁止。

## 证据与复现

- [英文完整报告](../report/report.md)、[本轮冻结协议](../research/proposal_completion_20261117/PROTOCOL.md)、[分析JSON](../research/proposal_completion_20261117/analysis/receipt.json)。
- [复现说明](../research/proposal_completion_20261117/REPRODUCE.md)、[失败及待解决项](../research/proposal_completion_20261117/UNRESOLVED.md)、[独立审查与裁定](../research/proposal_completion_20261117/review/RULINGS.md)。
- 图：`research/proposal_completion_20261117/figures/summary.png`，PDF/SVG同目录；源码、输入表、统计参数可追溯。版本化handoff路径记录于本轮handoff_receipt.json。

结论仅支持：测量可观测性、深度/对应完整性和裁判校准限制当前物理评价。不能据此断言生成模型物理正确率为0，也没有证明所有三维方法或所有VLM无效。
'''
    short = f'''# Results summary, 2026-11-17

Part A: {n_a} generated videos, no valid complete physical fits; g/T errors N/A. Original Part B: {n_b} production videos judged, calibration invalid, no valid ranking.

Fresh public inference: {d['videos']} DAVIS videos/{d['queries']} queries reproduce the frozen baseline; three PStudio examples extended to 150 frames. Co+depth has {all_invalid}/{all_visible} invalid required point-times; full EPE N/A. TAPIP3D full EPE macro mean {gm['tap']['raw_epe3d_m']*100:.2f} cm.

New local Qwen gates fail: gravity {gravity['correct']}/{gravity['calibration_cases']}, human-PC {pc['calibration_correct']}/{pc['calibration_planned']}; the separate balanced test is {pc['evaluation_correct']}/{pc['evaluation_planned']} diagnostic only. Generator expansion remains stopped. See [full report](report.md), [Chinese findings](../docs/PROPOSAL_RESULTS_2026-11-17.zh.md) and [raw failed leaderboard](../research/part_b_20261113T114040Z/run/results/leaderboard/part_b_leaderboard.json).
'''
    return {'report.md': english, 'PROPOSAL_RESULTS_2026-11-17.zh.md': zh, 'results_summary.md': short}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    for name, content in documents().items():
        with (args.output / name).open('x') as stream:
            stream.write(content)
    print(json.dumps({'output': str(args.output), 'documents': 3}))

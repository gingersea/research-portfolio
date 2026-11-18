# Reading the evidence

This portfolio preserves research history while giving interview readers a current interpretation. Original numerical files are copied unchanged or compressed losslessly. Only verified absolute Markdown links are made portable in the displayed Markdown copies; these edits are listed in `provenance/portable-link-changes.json`. The original interview PDFs are unchanged.

## Optimization project

- The intended two-factor comparison separates optimizer and parameter form. The actual 7B matrix is incomplete: the full-rank alternating branch fails, and the LoRA alternating branch omits ALS. It is not a fully controlled factorial causal estimate.
- Early OPT experiments used an incompatible tokenizer and unmasked padding labels. Later OPT schedule/probe/distillation results use a corrected tokenizer and label mask.
- Existing 7B main-matrix and SGD/A-SYNC scripts still clone token IDs into labels without masking padding. Historical values such as 1.25, 10.41, 6.82, and 6.83 are not promoted here as validated standard benchmark performance.
- The injection-timing finding is directly supported by the code sequence. Similar historical loss trajectories alone do not prove exact equivalence, and the 7B comparison must not be described as the corrected-timing injection experiment.
- Depth/amplification claims and simplified FLOPs multipliers in older documents are not universal validated laws or measured end-to-end speedups.
- For the HellaSwag summary, the current introduction consistently uses `downstream_eval.json`: approximately 56.72% for full-rank and 59.74% for LoRA, averaged over the three saved seeds. The full result is preserved as `downstream_eval.json.gz`; its small preview is adjacent. No new significance test is claimed.

Start with the [current project introduction](interview/01_优化与LoRA_完整介绍.md), then inspect the scripts and saved results. Earlier audits can describe files as missing even when later artifacts now exist; existence and scientific validity remain separate questions.

## Video project

- Zero valid complete fits among 24 generated videos means the measurement gate failed. It is not a claim that all videos violate physics.
- The original Part B calibration was invalid. Later success on narrowly controlled procedural scenes cannot retrospectively validate that ranking.
- New public-label visual-judge calibration also failed. Its test outputs remain diagnostic.
- Improved accepted precision can result from rejecting correct measurements. Report coverage and recovery alongside precision.
- The 325-input matrix contains several data sources, repeated analysis, and controlled variants. It is not 325 independent newly generated videos.
- The latest relocalization experiment used static MASt3R, not PointSt3R's unavailable fine-tuned weights.
- The action-conditioned pilot is a fully observed noiseless linear world. It demonstrates an interface and baseline, not a visual or robotic planning breakthrough.
- The current branch contains later work that was not committed in the original repository. This portfolio packages that working-tree snapshot; it does not rewrite the source repositories.

Start with the [current introduction](interview/02_视频物理一致性_完整介绍.md), [November 17 report](projects/world-model-part-b-20261113/docs/PROPOSAL_RESULTS_2026-11-17.zh.md), and [November 18 validation](projects/world-model-part-b-20261113/research/reference_relocalization_20261117/report_v2/REPORT.zh.md).

## Packaging validation

Packaging checks cover file integrity, compressed-result round trips, archive membership, primary navigation links, and credential-pattern scanning. They do not rerun training, video generation, scientific inference, or the archived test suites. Existing test counts refer to the dated source reports.

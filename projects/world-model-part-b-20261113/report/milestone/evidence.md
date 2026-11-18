# Milestone report evidence and requirements

## Requirements

The user supplied lecture 1 slide 79 verbatim: three pages; title/authors;
Introduction 20%, Related Work 25% (>=10 discussed citations), Data 15%,
Approach 20%, Preliminary Results 20%; actual training and analysis of at
least one deep learning model; next steps and obstacles. Reference pages are
included in the conservative three-page budget. The local original course
proposal is `/home/room115/jhy/4140/proposal.pdf`, identifying JIANG Huanyun,
1155233165, AIST4010 Foundation of Applied Deep Learning.

## Evidence map

- E01: `../../benchmark/results/raw/part_a_summary.json` (relative paths in final
  report text are repository-relative): 24 generated clips, Wan 11/12 tracked,
  CogVideoX 5/12 tracked, 0/24 accepted fits.
- E02: `benchmark/results/raw/detector_validation.json`: six synthetic checks;
  g errors 0.054%/0.162%, period error 0.138%.
- E03: `benchmark/results/training_pilot/metrics.json`: freshly executed
  supervised TinyUNet training, 29,465 parameters, 2048/256/256 synthetic
  train/validation/test frames, 12 epochs, Adam 0.001, batch 64, seed 7.
  Best epoch 12; held-out test IoU 0.9704117102, centroid error 0.1094954 px;
  color-rule baseline IoU 0.9014713509, centroid error 2.7697920 px.
- E04: `benchmark/training_pilot.py`: deterministic procedural split seeds,
  rendering, network, objective, validation-only model selection, metrics.
- E05: `benchmark/results/raw/part_a_audit.json`: all 24 videos decoded and
  metadata/hash/source checked. Frozen generator weights were not trained here.
- E06: `report/progress-2026-11-06.zh.md`: previous halfway status and limitations.

## Citation verification

ArXiv abstract pages/Atom records were opened by the AI assistant on 2026-11-07
and support the limited background claims used in this report. No human
bibliographic approval is claimed. Metadata are extracted programmatically.
All identifiers are in `identifiers.txt`.

Additional primary-source checks:
- Ho et al.: NeurIPS 2020 proceedings, paper hash 4c5bcfec8584af0d967f1ab10179ca4b.
- Rombach et al.: CVPR 2022 CVF proceedings, pp. 10684–10695.
- Peebles and Xie: ICCV 2023 CVF proceedings, pp. 4195–4205.
- Huang et al.: CVPR 2024 CVF proceedings, pp. 21807–21818.
- CogVideoX arXiv record explicitly states ICLR 2025 acceptance.
- Kang et al. arXiv record explicitly states ICML 2025.
- Adam arXiv record explicitly states ICLR 2015.
- Physics-IQ authors' project page supplies its WACV 2026 published citation.
- U-Net Crossref DOI 10.1007/978-3-319-24574-4_28 confirms MICCAI 2015,
  pp. 234–241, with matching authors/title.
- DBLP API requests were blocked by its bot protection; no DBLP verification
  is claimed. Official proceedings, author project pages, arXiv, and Crossref
  supply the bibliographic checks instead.

Correction to the earlier proposal: VideoPhy explicitly evaluates both open
and closed models (including CogVideoX); Kang already uses quantitative
simulation-based physics evaluation. Do not claim prior work only studies
closed models or that this is the first quantitative physics benchmark.

## Boundaries and choices

Training is an auxiliary segmentation pilot added to meet the course requirement
and address the observed measurement bottleneck. It is not fine-tuning Wan or
CogVideoX and was not applied to the archived 24-clip audit. Synthetic test scores
do not establish generated-video transfer. One training seed, no confidence
intervals or significance claims. Future work is explicitly labeled planned.
Part B has prompt/code scaffolding but no completed calibrated judging results.
The deadline is November 5 per the supplied rubric; experiments are not backdated.

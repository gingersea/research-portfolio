# Complete fixed framework matrix and traceable visual observations

Before new matrix scoring: all 144 existing heldout procedural cases x two
codecs =288 control video inputs, plus all previous30 DAVIS,3 PStudio,4 I2V.
No score-based selection and no new generation or threshold/model changes.
Expected control labels are loaded only AFTER numeric measurement, in evaluator.
The two scenes used for prior judge calibration are reported separately from
the remaining22. These data were evaluated by earlier scripts: this is full
framework equivalence/coverage validation, not a newly unseen model benchmark.

Scene is the independent procedural randomization unit (24, primary22), conditions
are within-scene, encodings are technical repeats. For each codec/condition show
correct count/22 and exact two-sided Clopper-Pearson95% interval; report paired
encoding disagreements, and all-errors/abstentions. No frame-level inference,
p-values or pooled independence over288. Compare raw measurements to previous
heldout_measurements_v2.json as an engineering regression check, not new GT.

Add optional visual-judgment import with exact opaque case/payload/sheet hashes,
full-sheet coverage and schema/admissibility validation. Keep numeric and visual
channels separate. Missing inputs remain pending; never invent votes or use
visual opinions as physical GT. Unknown/duplicate IDs and wrong hashes are hard
errors; missing IDs are explicitly reported, not silently counted as evaluated.

Current agent will inspect all20 already exported I2V sheets and provide first
context-aware AI diagnostic observations. Prior numerical/qualitative results
have been seen, so these are NOT independent blind labels. Record origin and
all actual viewed-image hashes; independence remains unverified/pending even if
their schema validates. This does not complete human annotation or independent
VLM accuracy verification. No new isolated judge agents are invoked.

Run325 once without opinions and once with validated opinions; only visual
channel may change. Preserve old outputs and source before changes, exact diff,
tests, comparison tables/figure, report and local-copy reproduction evidence.

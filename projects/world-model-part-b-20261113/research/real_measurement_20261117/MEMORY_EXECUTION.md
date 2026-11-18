# Memory-lifecycle correction, before human-label scoring

All12 first attempts remain in predictions/. Seven succeeded; five failed with
CUDA OOM, including973/1083-frame clips and subsequent shorter clips. No human
reference score has been computed when choosing this correction.

Tracebacks place failures in convolution/instance normalization and dense
temporal attention allocations, with large unused reserved memory. A fresh
process with the identical1083-frame first query and DEFAULT feature chunk200
succeeded at27,684,056,576 peak allocated bytes; chunk64 also succeeded at
27,533,935,104 bytes. Both receipts/predictions remain in memory_probes/.
Those probes show the case can fit natively; they do not alone prove every
allocation-lifecycle mechanism or prediction equivalence.

The single uniform correction is `torch.cuda.empty_cache()` immediately before
each full-video native query. It releases unused allocator cache and changes
neither frames, spatial resolution, dtype, weights, attention, feature chunk200,
queries/support grid, iteration count nor thresholds. No allocator environment
setting, model training or truncation is introduced. The source is separately
versioned as run_models_memory.py; all12 cases rerun to predictions_memory/.

Freeze runner/manifest hash and UTC before that rerun. Afterward compare arrays
for ALL7 successful first attempts, not selected examples. Record any coordinate,
probability or acceptance differences rather than assuming equivalence. Score
both attempts separately with the same evaluator. Primary report distinguishes
first-attempt execution failures from the corrected uniform run. Missing cases
remain in denominators in either run. Do not choose execution versions based on
human-label accuracy, and do not relabel failed calibration as passed.

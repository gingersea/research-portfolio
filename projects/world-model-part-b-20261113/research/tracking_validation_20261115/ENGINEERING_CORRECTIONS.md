# Engineering correction, before evaluation outputs were viewed

The initial clean_910 attempt under predictions/primary/development failed after
native inference: assert len(captured)==1. CoTrackerPredictor calls its child
model.forward(...) directly, so a PyTorch registered forward hook did not run.
This was an observation API failure, not a quality-based checkpoint/video retry.
No predictions from that attempt were selected or scored.

run_cotracker.py now wraps only that bound forward method, copies its returned
visibility/confidence for receipts, and returns the exact original tuple without
changing tensors or model parameters. Official source, checkpoint, RGB inputs,
queries, inference settings, thresholds and metrics remain unchanged. The original
runner is preserved in source_snapshots/pre_inference; corrected runs use the new
run-name observer_v2. New source snapshot precedes those runs. The old failed
receipt and development_inference.log remain intact.

## Dataset independence correction

After inference, an input-hash audit found that occlusion, crossing and
ambiguous_twins overwrote all seed-dependent geometry. Each of those conditions
had identical development and heldout videos. Those v1 results cannot count as
independent heldout validation. Their data, predictions and initial assessment
remain unchanged for audit; input hashes, not scores, triggered this correction.

The v2 renderer retains the already sampled geometric variation in these three
conditions (common shifts preserve the twins' indistinguishability). Other
conditions and all inference/metric thresholds remain identical. New data_v2,
data_manifest_v2.json, run-name dataset_v2 and results_v2 keep the revision explicit.
The generator asserts 30 distinct raw video pixel hashes before inference. The
new dataset/source snapshot is frozen before corrected inference. All30 sequences
are run with the same official checkpoint; none are selected based on quality.

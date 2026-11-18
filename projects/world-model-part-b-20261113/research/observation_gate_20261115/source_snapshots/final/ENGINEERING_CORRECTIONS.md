# Engineering history

No new heldout thresholds or model-quality retries are allowed. Initial inputs and
the frozen gate parameters remain unchanged.

1. Initial source transfer hit SSL EOF on a public source file; resumed with
   verified existing Git blobs, bounded retries, and four download workers.
   `fetch_tapip.log` retains the error; `fetch_tapip_resume.log` and
   `tapip_assets.json` record verified completion.
2. PyPI downloads timed out (proxy and direct pip attempts retained). Downloaded
   only hydra-core1.3.2 and python-box7.3.2 public wheels using PyPI metadata and
   checked SHA256; installed locally into the isolated environment. The base
   PyTorch environment is unchanged. Receipts: `environment_wheels.json` and
   `environment_*install*.log`.
3. Native CUDA build succeeded but direct import initially lacked the author's
   `pointops2` package alias (`ModuleNotFoundError`). Installed the unmodified
   author's local package with no dependencies/build isolation. Native KNN now
   matches CPU distances/indices; `prepare_run.log` retains the failure and
   `prepare_run_after_package_install.log` the successful check.
4. TAPIP3D's pretrained encoder constructor silently invokes mutable
   `torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")`
   before loading the complete TAPIP checkpoint. The first run stalled downloading
   `main.zip` and was interrupted BEFORE ANY VIDEO INFERENCE; full traceback is
   `tapip3d_primary.log`. The runner's one exact hub request now resolves to the
   already pinned, verified official CoTracker source/checkpoint. All other hub
   requests raise. Author TAPIP source and native kernels remain unchanged.
   Author loading uses strict=True; additionally EVERY final model state tensor
   is compared exactly to the TAPIP checkpoint after loading. This demonstrates
   that constructor dependency resolution does not change final learned weights.
   `run_model.before_hub_redirect.py` and `pre_inference_freeze.json` preserve the
   original runner; each actual prediction receipt records its runner hash and
   dependency resolution. The resumed log is `tapip3d_pinned_dependency.log`.

Missing basedpyright was previously declined; no LSP is installed. Actual Python,
native CUDA, inference, adversarial gate tests and result audits provide checks.
Scientific rejection/coverage failures must be reported separately from passing
software tests.

Native warnings retained: pointops uses deprecated CUDA typed-tensor constructors;
PyTorch's build helper has no compiler-version bounds for CUDA12.8. Compilation,
CPU-reference native KNN verification and all80 inferences succeeded. No warning
was suppressed. The copy command's -n portability warning is also retained in the
tool history; subsequent archival uses --update=none.

Figure v1 legend covered bars; v2 moved legends outside axes but its long right
axis label overlapped the supertitle. v3 removes the redundant shared-axis label
and shortens the remaining label. v1/v2 and pre-edit plotting sources are retained;
all three use identical scores, and auxiliary metrics are checked for exact
equality before re-rendering. This is a presentation-only correction.

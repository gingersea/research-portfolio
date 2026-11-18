# Preserved engineering history

- The official1.67GB DAVIS archive was initially fetched through the configured
  proxy. Direct hostname lookup failed; proxy transport was slow and produced
  intermittent TLS/proxy errors. A64worker attempt made no useful progress and
  was stopped; completed validated ranges were retained and concurrency restored.
- Public DNS-over-HTTPS supplied an A record for the SAME public GCS hostname.
  Direct HTTPS then worked with the original host/SNI, normal certificate
  verification, fixed object generation, exact range checks and final official
  MD5. No access control or authentication was bypassed; no data bytes changed.
  `fetch.log`, `fetch_resume64.log`, `fetch_resume24.log`, `fetch_direct.log` and
  before-source variants are retained. An attempted stop addressed an already
  nonexistent process and returned "No such process"; the actual recorded fetch
  process was then stopped. No unrelated process was modified.
- Three official3D sample downloads initially hit a TLS metadata error. Bounded
  retries and upstream generation/MD5 checks completed all3. Preserve
  `fetch_3d.log`, `fetch_3d_retry.log` and the pre-retry source.
- The authoritative3D samples use `tracks_XYZ` and `fx_fy_cx_cy`, unlike names in
  newer prose docs. Preparation follows actual published sample schema and checks
  initial query projection before inference. A read-only inspection using the
  prose key raised KeyError; no model/data result was overwritten.
- All650 DAVIS query inferences and both native models on all3 pilot cases
  completed on their first model attempt. No model-quality retries, scorer
  changes or post-score threshold tuning occurred.
- The official scorer's NumPy-only function bodies were extracted unchanged from
  pinned original source, avoiding TensorFlow-only reader installation. Original
  copyright/license header and attribution remain in the original source; the
  derived file is distributed with that source and NOTICE_TAPNET.md.
- mediapy1.2.7, a28KB wheel, was added only to a new isolated benchmark environment
  to use the official resize API. Prior study environments and model/source assets
  remain unchanged. The author's native pointops CUDA tensor-constructor warning
  remains in `run_3d.log`; no warning was suppressed.
- basedpyright/biome remain uninstalled following the previously stated preference.
  Actual inference, metric fixtures, source hashes, artifact audits and rendered
  figures provide the verification; no LSP pass is claimed.

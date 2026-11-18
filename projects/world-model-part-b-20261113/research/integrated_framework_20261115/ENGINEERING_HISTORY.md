# Integration history and remaining limitations

- The first focused collection intentionally failed because physcheck.relations
  did not yet exist. After implementation, seven boundary tests passed.
- The first 49-case replay completed with zero errors.
- An unconfigured CLI help probe omitted PYTHONPATH=benchmark and failed to find
  physcheck. The documented invocation includes PYTHONPATH and its help ran.
- fresh_i2v4_v1 failed before inference with TypeError: 'int' object is not
  iterable. TAPIP asset metadata's files field is a count; its actual source list
  is in tapip_source.json. Added the source_entries fixture (RED then GREEN),
  fixed only the adapter, and retained the first run directory. Fresh v2 then
  completed all four clips. No weights/native source/parameters changed.
- Self-review found initialization failures escaped before per-case accounting.
  A fixture reproduced RuntimeError: model not available (RED). Initialization
  failures now produce an explicit shared error and an error row for each
  affected I2V case, with no cache fallback (GREEN). Final replay49 and fresh4
  runs use that version; earlier runs remain retained. Final code review is
  author self-review, not an independent reviewer claim.
- The native pointops library emits its existing deprecated CUDA tensor
  constructor warning. It is retained; no shared native code was changed.
- basedpyright is unavailable and installation was previously declined. No LSP
  pass is claimed. Executed tests and real-run artifacts provide verification.
- Four generated clips still lack independent human labels and fresh visual
  judgments. Their visual packages are exported and marked pending. They receive
  numeric ABSTAIN rather than an invented physical failure or accuracy value.
- This is an integration deliverable. Dataset scores replay previous fixed
  predictions; it is not a new performance result or proof of physical validity.

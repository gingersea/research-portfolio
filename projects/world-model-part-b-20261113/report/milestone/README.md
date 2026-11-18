# AIST4010 milestone report

Deliverable: `milestone.pdf` — English, three A4 pages including references.
Author details come from the original course proposal. Requirements come from
lecture 1 slide 79, transcribed by the user.

The report includes 12 research works discussed in Related Work, plus one
software/workflow citation in the AI-assistance disclosure. Numeric macros are
generated from recorded experiments. The supervised TinyUNet pilot is a new,
actually executed experiment; Wan/CogVideoX remain frozen inference models.
Future work is explicitly distinguished from completed results.

From the worktree root:

```bash
python3 report/milestone/build_assets.py
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=report/milestone report/milestone/milestone.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=report/milestone report/milestone/milestone.tex
python3 report/milestone/verify_report.py
```

`verify_report.py` re-evaluates the saved trained checkpoint on regenerated
held-out test images, checks provenance hashes, PDF pages, required headings,
discussed-citation count, and LaTeX errors/overflow. It requires CUDA, PyMuPDF,
and the project's scientific Python environment. `evidence.md` records source
provenance, bibliography verification limits, and the scope of each claim.

Training evidence: `benchmark/results/training_pilot/metrics.json` and
`best_model.pt`. Reproduce training with `PYTHONPATH=benchmark python3
benchmark/training_pilot.py` (this replaces the pilot outputs).

"""Check the delivered PDF and independently re-evaluate the saved neural model."""
import hashlib
import json
from pathlib import Path
import re
import sys

import fitz
import numpy as np
import torch

here = Path(__file__).resolve().parent
root = here.parents[1]
sys.path.insert(0,str(root/"benchmark"))
from training_pilot import TinyUNet, make_dataset

record = json.loads((root/"benchmark/results/training_pilot/metrics.json").read_text())
checkpoint = root/"benchmark/results/training_pilot/best_model.pt"
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == record["model_sha256"]
assert hashlib.sha256((root/"benchmark/training_pilot.py").read_bytes()).hexdigest() == record["source_sha256"]
x,y,_ = make_dataset(256,seed=30001)
assert hashlib.sha256(x.numpy().tobytes()+y.numpy().tobytes()).hexdigest() == record["dataset_hashes"]["test"]
torch.set_num_threads(8)
model = TinyUNet().cuda().eval()
model.load_state_dict(torch.load(checkpoint,weights_only=True))
assert sum(p.numel() for p in model.parameters()) == record["parameter_count"]
with torch.no_grad():
    predicted = torch.cat([model(batch.cuda()).sigmoid().cpu() for batch in x.split(64)])[:,0].numpy() > .5
truth = y[:,0].numpy().astype(bool)
ious,errors = [],[]
for p,t in zip(predicted,truth):
    ious.append(np.logical_and(p,t).sum()/np.logical_or(p,t).sum())
    py,px = np.where(p)
    ty,tx = np.where(t)
    errors.append(np.linalg.norm([px.mean()-tx.mean(),py.mean()-ty.mean()]) if len(px) else np.hypot(*p.shape))
assert np.isclose(np.mean(ious),record["test"]["mean_iou"],atol=1e-8)
assert np.isclose(np.mean(errors),record["test"]["centroid_error_px"],atol=1e-8)

pdf = fitz.open(here/"milestone.pdf")
assert len(pdf) == 3
text = "\n".join(page.get_text() for page in pdf)
for heading in ["1. Introduction","2. Related Work","3. Data","4. Approach","5. Preliminary Results"]:
    assert heading in text, heading
assert "JIANG Huanyun" in text and "1155233165" in text
assert "??" not in text
for key in ("mean_iou","centroid_error_px"):
    assert f"{record['test'][key]:.3f}" in text
for page in pdf:
    for block in page.get_text("blocks"):
        assert fitz.Rect(block[:4]) in page.rect, "out-of-page text"
tex = (here/"milestone.tex").read_text()
related = tex.split(r"\sect{2}{Related Work}")[1].split(r"\newpage")[0]
keys = {key for group in re.findall(r"\\cite\{([^}]+)\}",related) for key in group.split(",")}
assert len(keys) >= 10
log = (here/"milestone.log").read_text()
assert "undefined" not in log.lower() and "Overfull" not in log
validation = json.loads((here/"citation_validation.json").read_text())
result = {"pdf_pages":len(pdf),"related_work_distinct_citations":len(keys),
          "test_iou_reproduced":float(np.mean(ious)),"test_centroid_error_reproduced":float(np.mean(errors)),
          "checkpoint_hash_matches":True,"source_and_test_data_hashes_match":True,
          "all_five_sections":True,"no_unresolved_references":True,"no_overfull_boxes":True,
          "pdf_sha256":hashlib.sha256((here/"milestone.pdf").read_bytes()).hexdigest()}
(here/"verification.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result,indent=2))

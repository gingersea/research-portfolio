"""Explain real-domain rejection; diagnostic only, no rule/label changes."""
from collections import Counter
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('frozen_gate_diagnostic',ROOT.parent/'observation_gate_20261115/gate.py')
gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)


def main():
    rows=[]
    for case in json.loads((ROOT/'real_manifest.json').read_text()):
        data=np.load(case['source']);rgb=data[data.files[0]]
        patches=[cv2.getRectSubPix(rgb[0].astype(np.float32),(33,33),tuple(map(float,c))) for c in case['initial_centers']]
        gray=[cv2.resize(cv2.cvtColor(p,cv2.COLOR_RGB2GRAY),(25,25)) for p in patches]
        row={'clip':case['clip'],'gray_initial_reference_similarity':gate.similarity(gray[0],gray[1]),
             'initial_RGB_mean_color_distance':float(np.linalg.norm(patches[0].mean(axis=(0,1))-patches[1].mean(axis=(0,1)))),
             'gray_unknown_latch_threshold':.95,'rules_changed_after_real_result':False,'reason_counts':{}}
        for model in ['cotracker','tapip3d']:
            details=json.loads((ROOT/'real_depth_results'/f'{case["clip"]}_{model}_reasons.json').read_text())
            row['reason_counts'][model]=dict(Counter(item['reason'] for frame in details for item in frame))
        rows.append(row)
    path=ROOT/'real_depth_results/appearance_rejection_diagnosis.json';assert not path.exists()
    path.write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows,indent=2))


if __name__=='__main__':main()

"""Reuse verified native runner, changing only derived study root and model flag."""
import importlib.util
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
path=ROOT.parent/'observation_gate_20261115/run_model.py'
sys.path.insert(0,str(path.parent))
spec=importlib.util.spec_from_file_location('native_runner',path)
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
runner.ROOT=ROOT
sys.argv=[sys.argv[0],'--model','cotracker',*sys.argv[1:]]
runner.main()

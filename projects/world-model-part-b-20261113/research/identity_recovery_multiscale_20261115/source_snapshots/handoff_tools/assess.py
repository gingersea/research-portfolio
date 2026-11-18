"""Same fixed V1 metric/event definitions, separately derived V2 predictions."""
import importlib.util
from pathlib import Path
import sys

from recovery import recover

ROOT=Path(__file__).resolve().parent
path=ROOT.parent/'identity_recovery_20261115/assess_controls.py';sys.path.insert(0,str(path.parent))
spec=importlib.util.spec_from_file_location('fixed_assessment',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.ROOT=ROOT;m.recover=recover;m.main()

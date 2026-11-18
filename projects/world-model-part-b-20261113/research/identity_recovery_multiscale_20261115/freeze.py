"""Reused source/dependency freezing with a separate study root."""
import importlib.util
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
path=ROOT.parent/'identity_recovery_20261115/freeze.py';sys.path.insert(0,str(path.parent))
spec=importlib.util.spec_from_file_location('freezer',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.ROOT=ROOT;m.main()

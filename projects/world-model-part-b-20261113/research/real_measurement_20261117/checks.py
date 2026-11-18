"""Capture tests and compile results; no empirical score changes."""
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]

if __name__=='__main__':
    tests=subprocess.run([sys.executable,'-m','pytest','-q'],cwd=PROJECT,capture_output=True,text=True)
    compiled=subprocess.run([sys.executable,'-m','compileall','-q',str(ROOT),str(PROJECT/'benchmark/physcheck/relations/sparse.py')],
                            cwd=PROJECT,capture_output=True,text=True)
    record={'command':[sys.executable,'-m','pytest','-q'],'exit':tests.returncode,'stdout':tests.stdout,'stderr':tests.stderr,
            'compile_exit':compiled.returncode,'compile_stderr':compiled.stderr,'python':platform.python_version(),
            'packages':{name:importlib.metadata.version(name) for name in ['numpy','torch','opencv-python','pytest','matplotlib']}}
    with (ROOT/'tests_receipt.json').open('x') as stream:
        json.dump(record,stream,indent=2)
    print(tests.stdout)
    raise SystemExit(tests.returncode or compiled.returncode)

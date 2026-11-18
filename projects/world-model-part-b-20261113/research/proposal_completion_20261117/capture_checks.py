"""Record the final actual test result and installed numerical runtime."""
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]

if __name__ == '__main__':
    result = subprocess.run([sys.executable, '-m', 'pytest', '-q'], cwd=PROJECT, text=True, capture_output=True)
    record = {'command': [sys.executable, '-m', 'pytest', '-q'], 'cwd': str(PROJECT),
              'exit': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr,
              'python': platform.python_version(), 'packages': {}}
    for name in ['numpy', 'torch', 'transformers', 'opencv-python', 'matplotlib', 'pytest']:
        try:
            record['packages'][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            record['packages'][name] = None
    with (ROOT / 'tests_final.json').open('x') as stream:
        json.dump(record, stream, indent=2)
    print(result.stdout)
    raise SystemExit(result.returncode)

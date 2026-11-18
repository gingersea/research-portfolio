"""Verify the archived study by relative paths after copying or extraction."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('study',type=Path,nargs='?',default=Path(__file__).resolve().parent)
    args=parser.parse_args()
    study=args.study.resolve()
    checksums=json.loads((study/'checksums.json').read_text())
    failures=[]
    for relative,expected in checksums.items():
        path=(study/relative).resolve()
        if not path.is_relative_to(study) or not path.is_file():
            failures.append(relative)
            continue
        with path.open('rb') as stream:
            actual=hashlib.file_digest(stream,'sha256').hexdigest()
        if actual!=expected:
            failures.append(relative)
    if failures:
        raise SystemExit('Snapshot checksum failures: '+', '.join(failures))
    results=study/'run/results_reviewed'
    if not results.exists():
        results=study/'run/results'
    board=json.loads((results/'leaderboard/part_b_leaderboard.json').read_text())
    print(json.dumps({'checksums_passed':len(checksums),'calibration_valid':board['valid'],
                      'verified_root':str(study)},indent=2))


if __name__=='__main__':main()

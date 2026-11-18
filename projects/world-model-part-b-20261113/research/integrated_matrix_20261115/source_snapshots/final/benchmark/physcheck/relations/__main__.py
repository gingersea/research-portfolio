"""PYTHONPATH=benchmark python -m physcheck.relations --manifest ... --output ..."""
import argparse
import json
from pathlib import Path

from .pipeline import run


def main():
    parser = argparse.ArgumentParser(description='Run the fixed integrated measurement framework')
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--domain', choices=['control', 'davis', 'pstudio', 'i2v'])
    parser.add_argument('--fresh-i2v', action='store_true')
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--visual-judgments', type=Path, help='JSON array of evidence-bound diagnostic visual opinions')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if args.domain:
        manifest['cases'] = [c for c in manifest['cases'] if c['domain'] == args.domain]
    opinions = json.loads(args.visual_judgments.read_text()) if args.visual_judgments else None
    summary = run(manifest, args.output, args.fresh_i2v, args.gpu, opinions)
    print(json.dumps(summary, indent=2))
    if summary['errors']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

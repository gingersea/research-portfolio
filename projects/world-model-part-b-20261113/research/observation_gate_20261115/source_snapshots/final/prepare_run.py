"""Freeze operating points/source receipts before new inference; native CUDA sanity."""
import json
import platform
from pathlib import Path
import sys

from fetch_tapip import sha

ROOT = Path(__file__).resolve().parent


def main():
    grid = json.loads((ROOT/'calibration/grid.json').read_text())['cases']
    eligible = [r for r in grid if r['unsupported'] == 0 and
                r['accepted_correct']/r['accepted_total'] >= .99]
    secondary = sorted(eligible, key=lambda r: (-r['accepted_correct'], -r['config']['identity_margin']))[0]
    points = {'primary': json.loads((ROOT/'gate_config.json').read_text()),
              'coverage': secondary['config']}
    assert points['coverage'] == {'aperture': .5, 'correlation': .8, 'identity_margin': .15}
    for name, value in [('gate_operating_points.json', points)]:
        target = ROOT/name
        if target.exists():
            assert json.loads(target.read_text()) == value
        else:
            target.write_text(json.dumps(value, indent=2)+'\n')
    import torch
    import hydra
    import box
    assets = json.loads((ROOT/'tapip_assets.json').read_text())
    code = Path(assets['source_directory'])
    sys.path[:0] = [str(code), str(code/'third_party/pointops2')]
    from third_party.pointops2.functions.pointops import knnquery
    torch.cuda.set_device(1)
    generator = torch.Generator().manual_seed(55)
    xyz = torch.randn(61, 3, generator=generator).cuda()
    query = torch.randn(7, 3, generator=generator).cuda()
    indices, distances = knnquery(5, xyz, query, torch.tensor([61],device='cuda',dtype=torch.int32),
                                 torch.tensor([7],device='cuda',dtype=torch.int32))
    expected, expected_indices = torch.cdist(query.cpu(), xyz.cpu()).topk(5, largest=False)
    assert torch.equal(indices.cpu().long(), expected_indices)
    assert torch.allclose(distances.cpu(), expected, atol=1e-5)
    record = {'python': platform.python_version(), 'torch': torch.__version__,
              'cuda': torch.version.cuda, 'hydra': hydra.__version__, 'box': box.__version__,
              'native_knn_cpu_reference': 'passed: 7queries x5 neighbors, 61points',
              'native_extension_sha256': sha(code/'third_party/pointops2/pointops2_cuda.cpython-312-x86_64-linux-gnu.so'),
              'files': {str(p.relative_to(ROOT)): sha(p) for p in ROOT.iterdir()
                        if p.suffix in ['.py', '.md', '.json'] and p.name != 'pre_inference_freeze.json'}}
    target = ROOT/'pre_inference_freeze.json'
    if target.exists():
        raise FileExistsError(target)
    target.write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()

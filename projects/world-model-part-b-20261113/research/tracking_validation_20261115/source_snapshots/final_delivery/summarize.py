"""Observed result plots and provenance checks; no threshold/model tuning."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

from fetch_assets import sha
from render_controls import CONDITIONS

ROOT = Path(__file__).resolve().parent
METHODS = ['legacy_hue', 'foreground_v1', 'cotracker_official', 'cotracker_vc09']
COLORS = ['#888888', '#e69f00', '#0072b2', '#009e73']


def save(fig, path, top=.95):
    fig.tight_layout(rect=(0,0,1,top))
    for extension in ['png', 'pdf', 'svg']:
        fig.savefig(path.with_suffix('.'+extension), dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='figures')
    args = parser.parse_args()
    output = ROOT/args.output
    output.mkdir(exist_ok=False)
    plt.rcParams.update({'font.size': 8, 'svg.fonttype': 'none', 'pdf.fonttype': 42})
    results = json.loads((ROOT/'results_v2/two_d.json').read_text())
    heldout = [row for row in results['condition_scores'] if row['split']=='heldout']
    fig, axes = plt.subplots(2, 2, figsize=(15, 8))
    for axis, metric, label in zip(axes.flat,
            ['pck3_visible','visible_recall','false_visible_rate','wrong_other_claim_fraction'],
            ['Visible identifiable PCK at3px','Visibility recall','False-visible fraction among GT-invisible samples','Wrong-other-center claims / declared known-identity samples']):
        for i, (method, color) in enumerate(zip(METHODS, COLORS)):
            values = [next(row[metric] for row in heldout if row['condition']==condition and row['method']==method) for condition in CONDITIONS]
            values = [np.nan if value is None else value for value in values]
            axis.bar(np.arange(10)+(i-1.5)*.19, values, width=.19, color=color, label=method)
        axis.set(xticks=range(10), xticklabels=CONDITIONS, ylim=(0,1.08), ylabel='Fraction', title=label)
        axis.tick_params(axis='x', labelrotation=45)
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5,.955), ncol=4, fontsize=8)
    fig.suptitle('Heldout controls: 2 sequences/condition, frame0 excluded; empty negative denominators are N/A')
    save(fig, output/'heldout_2d_metrics', top=.91)
    geometry = json.loads((ROOT/'results_v2/geometry.json').read_text())
    variants = ['constant_depth_4m','scene_gt_depth','gt_depth_temporal_10percent','gt_depth_scale_bias_20percent','gt_depth_no_camera_compensation']
    names = ['Constant4m','Scene GT depth','10% temporal scale','20% scale bias','No camera compensation']
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for axis, condition in zip(axes, ['clean','camera_motion','identity_replacement']):
        for i, variant in enumerate(variants):
            values = [100*row['world_epe_m'] for row in geometry['rows'] if row['split']=='heldout' and row['condition']==condition and row['variant']==variant and row['policy']=='all_gt_visible']
            axis.bar(i, np.mean(values), color='#56b4e9')
            axis.scatter(np.full(len(values), i), values, s=18, c='black', zorder=3)
        axis.set(xticks=range(5), xticklabels=names, ylabel='World position EPE (cm)', title=condition)
        axis.tick_params(axis='x', labelrotation=50)
    fig.suptitle('Same CoTracker XY, privileged geometry inputs; dots=sequences (n=2), bars=means; different panel scales')
    save(fig, output/'geometry_information_ablation')
    fig, axes = plt.subplots(10, 5, figsize=(14, 17))
    manifest = json.loads((ROOT/'data_manifest_v2.json').read_text())
    for row_index, condition in enumerate(CONDITIONS):
        case = next(row for row in manifest['cases'] if row['condition']==condition and row['seed']==1910)
        inputs = np.load(ROOT/case['input'], allow_pickle=False)
        truth = np.load(ROOT/case['truth'], allow_pickle=False)
        prediction = np.load(ROOT/'predictions/dataset_v2/heldout'/f'{case["case"]}.npz', allow_pickle=False)
        for column, frame in enumerate([1, 12, 16, 24, 32]):
            axis = axes[row_index, column]
            axis.imshow(inputs['video'][frame])
            for obj, query in enumerate([0,5]):
                gx, gy = truth['xy'][frame, query]
                px, py = prediction['xy'][frame, query]
                visible = bool(prediction['official_visible'][frame, query])
                axis.add_patch(Circle((gx,gy), 3, fill=False, color='lime' if truth['visible'][frame, query] else 'black', linewidth=.8))
                axis.plot(px,py,'x',color='red' if obj==0 else 'blue', markersize=4, alpha=1 if visible else .25)
            axis.set_title(f'{condition} f{frame}', fontsize=7)
            axis.set_xlim(0,383); axis.set_ylim(255,0); axis.axis('off')
    fig.suptitle('One heldout sequence/condition: rings=GT3px, x=predicted centers; faint x=not declared visible; frame0 excluded')
    save(fig, output/'heldout_observation_overlay', top=.975)
    frozen = json.loads((ROOT/'source_snapshots/dataset_v2/manifest.json').read_text())
    assert all(sha(ROOT/name)==digest for name,digest in frozen['sha256'].items())
    assert all(sha(ROOT.parent.parent/name)==digest for name,digest in frozen['read_only_baseline_sha256'].items())
    assets = json.loads((ROOT/'assets.json').read_text())
    assert sha(Path(assets['checkpoint'])) == assets['checkpoint_sha256']
    assert all(sha(Path(assets['source_directory'])/item['file'])==item['sha256'] for item in assets['source_files'])
    assert len({row['video_pixel_sha256'] for row in manifest['cases']})==30
    native_records = []
    for case in manifest['cases']:
        folder = ROOT/'predictions/dataset_v2'/case['split']
        record = json.loads((folder/f'{case["case"]}.json').read_text())
        assert record['status']=='ok' and record['input_sha256']==case['input_sha256']
        assert record['data_manifest_sha256']==sha(ROOT/'data_manifest_v2.json')
        assert record['protocol_sha256']==sha(ROOT/'PROTOCOL.md')
        assert sha(folder/f'{case["case"]}.npz') == record['prediction_sha256']
        native_records.append(record)
    old_manifest = json.loads((ROOT/'data_manifest.json').read_text())
    old_cases = {case['case']:case for case in old_manifest['cases']}
    replays = []
    for case in manifest['cases']:
        if old_cases[case['case']]['input_sha256'] != case['input_sha256']:
            continue
        a = np.load(ROOT/'predictions/observer_v2'/case['split']/f'{case["case"]}.npz', allow_pickle=False)
        b = np.load(ROOT/'predictions/dataset_v2'/case['split']/f'{case["case"]}.npz', allow_pickle=False)
        replays.append({'case': case['case'], 'maximum_coordinate_difference_px': float(np.max(np.abs(a['xy']-b['xy'])))})
    record = {'final_native_sequences':len(native_records),'final_frames':30*33,'main_center_queries_per_sequence':2,
              'auxiliary_material_queries_per_sequence':10, 'unique_final_video_pixel_hashes':30,
              'development_sequences':10,'heldout_sequences':20,'evaluated_post_query_frames_per_sequence':32,
              'v1_duplicate_conditions':['occlusion','crossing','ambiguous_twins'],'v1_is_not_final_validation':True,
              'frozen_source_files_unchanged':len(frozen['sha256']), 'baseline_sources_unchanged':True,
              'native_source_files_verified':len(assets['source_files']), 'checkpoint_verified':True,
              'unchanged_input_replay_checks':replays,'maximum_replay_difference_px':max(row['maximum_coordinate_difference_px'] for row in replays),
              'geometry_exact_query_error_m':geometry['exact_query_max_error_m'],
              'maximum_geometry_reprojection_change_px':max(row['projection_change_from_2d_px'] for row in geometry['rows']),
              'gates':results['gates'],'full_learned_3d_model_run':False,'python':platform.python_version(),
              'versions':{name:importlib.metadata.version(name) for name in ['torch','numpy','scipy','opencv-python','matplotlib','Pillow','einops','pytest','requests']}}
    verification_path = ROOT/('verification.json' if args.output=='figures' else f'verification_{args.output}.json')
    assert not verification_path.exists(), 'Refuse to replace verification evidence'
    verification_path.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    print(json.dumps({key:value for key,value in record.items() if key!='unchanged_input_replay_checks'},indent=2))


if __name__ == '__main__':
    main()

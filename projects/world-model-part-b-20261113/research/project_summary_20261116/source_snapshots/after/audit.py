"""Documentation-only source catalogue, consistency checks and local handoff."""
from datetime import datetime, timezone
import difflib
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]

STUDIES = [
    ('transfer_20261109T143307Z', 'Transfer screening', 'complete_unlabeled', 'REPORT.zh.md'),
    ('stress_20261109T144046Z', 'Scale stress and sampling fixes', 'complete_scoped_controls', 'REPORT.zh.md'),
    ('multiscale_20261109T191120Z', 'Matched multiscale training', 'primary_endpoint_not_improved', 'REPORT.zh.md'),
    ('part_b_20261113T114040Z', 'Part B production and isolated judgments', 'production_complete_calibration_invalid', 'REPORT.md'),
    ('closeout_20261114', 'Sparse-frame information limit', 'complete_scoped_counterexamples', 'results/REPORT.md'),
    ('action_world_stage0_20261114', 'Action-conditioned linear-state pilot', 'complete_toy_side_study', 'results/REPORT.md'),
    ('physics_methods_20261114', 'Literature, public labels and VPT trial', 'complete_scoped_trial', 'REPORT.zh.md'),
    ('principia_gravity_20261114', 'Controlled two-object gravity relation', 'scoped_calibration_passed', 'REPORT.zh.md'),
    ('principia_i2v_20261115', 'Native 2-state x 2-seed I2V', 'generation_complete_measurement_gate_failed', 'REPORT.zh.md'),
    ('tracking_audit_20261115', 'Coverage audit and identity counterexample', 'identity_challenge_failed', 'REPORT.zh.md'),
    ('tracking_validation_20261115', 'Native 2D tracking and geometry ablation', 'baseline_complete_production_gate_failed', 'REPORT.zh.md'),
    ('observation_gate_20261115', 'Independent RGB gate and native TAPIP3D', 'scoped_gate_complete_recovery_gap', 'REPORT.zh.md'),
    ('identity_recovery_20261115', 'Recovery V1, I2V reference and depth chain', 'mixed_scoped_results_real_gap', 'REPORT.zh.md'),
    ('identity_recovery_multiscale_20261115', 'Recovery V2 on new controls', 'scoped_positive_result_not_real_validation', 'REPORT.zh.md'),
    ('paper_benchmark_20261115', 'DAVIS and calibrated PStudio pilot', 'external_label_baselines_complete', 'REPORT.zh.md'),
    ('measurement_validation_20261115', 'Acceptance calibration and error attribution', 'precision_coverage_tradeoff_not_general_fix', 'REPORT.zh.md'),
    ('integrated_framework_20261115', 'Unified entry and fresh I2V chain', 'engineering_complete_scientific_gaps', 'REPORT.zh.md'),
    ('integrated_matrix_20261115', '325-input matrix and diagnostic opinions', 'engineering_equivalence_complete', 'REPORT.zh.md'),
]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write('\n')


def load(relative):
    return json.loads((PROJECT / relative).read_text())


def catalogue():
    records, bindings = [], {}
    for directory, name, status, report in STUDIES:
        folder = PROJECT / 'research' / directory
        path = folder / report
        assert path.is_file(), path
        bindings[str(path.relative_to(PROJECT))] = sha(path)
        handoff = folder / 'handoff_receipt.json'
        records.append({'directory': directory, 'work': name, 'status': status,
             'report': str(path.relative_to(PROJECT)), 'report_sha256': sha(path),
             'handoff_receipt': str(handoff.relative_to(PROJECT)) if handoff.exists() else None,
             'provenance_role': 'historical evidence at its own cutoff; later reports can supersede next-step statements'})
    evidence = ['benchmark/results/raw/part_a_summary.json', 'benchmark/results/raw/part_a_audit.json',
         'benchmark/results/training_pilot/metrics.json', 'research/physics_methods_20261114/generation_audit.json',
         'research/physics_methods_20261114/videophy2_analysis.json',
         'research/principia_gravity_20261114/heldout_gate_v2.json',
         'research/principia_gravity_20261114/judge_gate_v2.json', 'research/principia_i2v_20261115/pilot_gate.json',
         'research/identity_recovery_multiscale_20261115/control_results/summary.json',
         'research/paper_benchmark_20261115/data_manifest.json', 'research/paper_benchmark_20261115/results/summary.json',
         'research/paper_benchmark_20261115/pilot3d_results/summary.json',
         'research/measurement_validation_20261115/results/summary.json',
         'research/measurement_validation_20261115/geometry_results/summary.json',
         'research/integrated_framework_20261115/fresh_i2v4_final/summary.json',
         'research/integrated_matrix_20261115/analysis/summary.json',
         'research/integrated_matrix_20261115/tests_receipt.json',
         'research/integrated_matrix_20261115/fault_verification.json',
         'research/integrated_matrix_20261115/handoff_receipt.json']
    for name in evidence:
        bindings[name] = sha(PROJECT / name)
    write_json(ROOT / 'evidence_catalog.json', {'cutoff_local_date': '2026-11-16', 'dated_studies': records,
         'additional_legacy_work': 'Part A and detector pilot live under benchmark/results and report/',
         'historical_sources_sha256': bindings, 'no_experiment_count_addition_across_replays': True})
    lines = ['# 历次研究与证据索引', '', '当前整体状态与路线见[项目总结](../../docs/PROJECT_SUMMARY_2026-11-16.zh.md)。',
         '这里列出18个既有研究目录；表内状态只表示已执行证据的范围，不是新增实验或通用有效性认证。',
         '早期Part A和检测器试点另存于benchmark/results与report，不计入这18目录。', '',
         '|研究目录|工作|状态/范围|原报告|', '|---|---|---|---|']
    for row in records:
        path = Path('..') / row['directory'] / Path(row['report']).relative_to(Path('research') / row['directory'])
        lines.append(f'|{row["directory"]}|{row["work"]}|{row["status"]}|[报告]({path})|')
    lines += ['', '状态中的failed/invalid均保留为实测失败，scoped不代表已完成自然域验证。',
         '已查看的DAVIS/控制不能作为调参后的全新留出；编码、点、帧和重复运行不作为新增独立样本。',
         '目录内handoff_receipt.json（若有）指向版本化共享交接。source_snapshots保留对应阶段代码。',
         '完整机器来源和SHA256见[evidence_catalog.json](evidence_catalog.json)。']
    (ROOT / 'EXPERIMENT_INDEX.md').write_text('\n'.join(lines) + '\n')
    return bindings


def validate_numbers():
    part = load('benchmark/results/raw/part_a_summary.json')['by_model']
    assert sum(m['n_videos'] for m in part.values()) == 24
    assert all(m['fit_success_rate'] == 0 for m in part.values())
    assert part['wan2.1_1.3b']['track_status_counts']['ok'] == 11
    assert part['cogvideox_5b']['track_status_counts']['ok'] == 5
    trial = load('research/physics_methods_20261114/generation_audit.json')
    assert trial['production_videos'] == 6 and not trial['physical_improvement_claim']
    gate = load('research/principia_gravity_20261114/judge_gate_v2.json')
    assert gate['judge_passed'] and gate['validated_cases'] == 12
    i2v = load('research/principia_i2v_20261115/pilot_gate.json')
    assert i2v['planned_clips'] == 4 and i2v['initial_all_passed'] and not i2v['expand_generation']
    v2 = load('research/identity_recovery_multiscale_20261115/control_results/summary.json')
    r = next(r for r in v2 if r['split'] == 'heldout' and r['condition'] == 'ALL' and r['method'] == 'recovery')
    assert r['correct'] == r['accepted'] == 1825 and r['recovered_events'] == r['events'] == 15
    native = next(r for r in load('research/paper_benchmark_20261115/results/summary.json') if r['policy'] == 'native')
    assert math.isclose(native['average_jaccard'] * 100, 60.56, abs_tol=.005)
    cal = next(r for r in load('research/measurement_validation_20261115/results/summary.json')
               if r['split'] == 'validation' and r['policy'] == 'calibrated')
    assert round(cal['precision'] * 100, 2) == 97.26 and round(cal['retention'] * 100, 2) == 38.41
    current = load('research/integrated_matrix_20261115/analysis/summary.json')
    assert current['total_framework_inputs'] == 325 and current['framework_errors'] == 0
    assert current['control_verdicts'] == {'PASS': 48, 'FAIL': 48, 'ABSTAIN': 192}
    assert current['noncalibration_scenes'] == 22 and current['i2v_independent_visual_pending'] == 4
    tests = load('research/integrated_matrix_20261115/tests_receipt.json')
    assert tests['exit'] == 0 and '98 passed' in tests['stdout']
    return {'legacy_part_a': '24 videos; zero valid complete fits', 'vpt_trial': '6 clips, not physical improvement evidence',
         'gravity_judge': '12 scoped valid judgments', 'i2v': '4 initial states retained, expansion gate failed',
         'recovery_v2': 'scoped15/15 events, not real-video validation', 'davis_AJ': native['average_jaccard'],
         'calibrated_validation_precision': cal['precision'], 'calibrated_validation_retention': cal['retention'],
         'latest_matrix': current, 'tests': 'historical98 pass receipt checked; no new suite run'}


def check_links():
    documents = [PROJECT / 'README.md', PROJECT / 'docs/PROJECT_SUMMARY_2026-11-16.zh.md', ROOT / 'EXPERIMENT_INDEX.md']
    checked = []
    for document in documents:
        for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)', document.read_text()):
            target = target.strip('<>')
            if urlparse(target).scheme or target.startswith('#'):
                continue
            local = unquote(target.split('#', 1)[0])
            path = (document.parent / local).resolve()
            assert path.exists(), (document, target)
            checked.append({'document': str(document.relative_to(PROJECT)), 'target': target})
    return checked


def finish(bindings, checks):
    before = ROOT / 'source_snapshots/before/README.md'
    after = ROOT / 'source_snapshots/after'
    after.mkdir(exist_ok=False)
    paths = [PROJECT / 'README.md', PROJECT / 'docs/PROJECT_SUMMARY_2026-11-16.zh.md',
             ROOT / 'EXPERIMENT_INDEX.md', ROOT / 'audit.py']
    for path in paths:
        shutil.copy2(path, after / path.name)
    diff = ''.join(difflib.unified_diff(before.read_text().splitlines(keepends=True),
                 (PROJECT / 'README.md').read_text().splitlines(keepends=True), fromfile='before/README.md', tofile='after/README.md'))
    (after / 'README.exact.patch').write_text(diff)
    for name, digest in bindings.items():
        assert sha(PROJECT / name) == digest, name
    write_json(ROOT / 'verification.json', {'status': 'passed', 'scope': 'source consistency, claims and local Markdown links only',
         'utc': datetime.now(timezone.utc).isoformat(), 'historical_study_reports': 18,
         'unchanged_historical_evidence_files': len(bindings), 'checks': checks,
         'local_links': check_links(), 'README_before_sha256': sha(before),
         'README_after_sha256': sha(PROJECT / 'README.md'), 'new_training_inference_or_evaluation': False,
         'new_test_suite_run': False, 'remote_or_GitHub_checked_or_updated': False})
    bundle = Path('/data/AI_Research/shared/handoff') / ('world_model_project_summary_20261116_v1_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    bundle.mkdir()
    shutil.copytree(ROOT, bundle / 'research/project_summary_20261116')
    (bundle / 'docs').mkdir()
    shutil.copy2(PROJECT / 'README.md', bundle / 'README.md')
    shutil.copy2(PROJECT / 'docs/PROJECT_SUMMARY_2026-11-16.zh.md', bundle / 'docs/PROJECT_SUMMARY_2026-11-16.zh.md')
    # A documentation handoff: copy referenced reports/compact evidence, not raw media/weights.
    for name in bindings:
        source = PROJECT / name
        target = bundle / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        assert sha(source) == sha(target)
    files = {str(p.relative_to(bundle)): sha(p) for p in bundle.rglob('*') if p.is_file()}
    write_json(bundle / 'MANIFEST.json', {'files_sha256': files,
         'handoff_scope': 'summary/index and frozen report/compact-evidence copies; not a new executable experimental bundle',
         'full_experiment_data': 'original dated study handoffs and shared paths'})
    write_json(ROOT / 'handoff_receipt.json', {'path': str(bundle), 'copied_files': len(files), 'source_copies_verified': True})
    print(json.dumps({'historical_studies': 18, 'historical_files_unchanged': len(bindings),
                     'local_links_checked': len(check_links()), 'handoff': str(bundle)}))


if __name__ == '__main__':
    finish(catalogue(), validate_numbers())

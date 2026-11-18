"""Report tables preserve unavailable physics and invalid calibration explicitly."""


def render_part_a(summary):
    lines = ['| Model | Videos | Tracked | Valid complete fits | Mean g relative error | Mean T relative error |',
             '|---|---:|---:|---:|---:|---:|']
    for name, row in sorted(summary['by_model'].items()):
        n = row['n_videos']
        g = 'N/A' if row['g_rel_err_mean'] is None else f'{row["g_rel_err_mean"]:.4f}'
        t = 'N/A' if row['T_rel_err_mean'] is None else f'{row["T_rel_err_mean"]:.4f}'
        tracked = round(row['track_success_rate'] * n)
        fitted = round(row['fit_success_rate'] * n)
        lines.append(f'| {name} | {n} | {tracked}/{n} | {fitted}/{n} | {g} | {t} |')
    return '\n'.join(lines)


def render_part_b(summary):
    valid = summary['valid']
    data = summary if valid else summary['diagnostic_only']
    lines = ['| Model | Judged production clips | Valid violation rate |', '|---|---:|---|']
    for name in sorted(data['violation_rate_by_model']):
        n = sum(r['model'] == name for r in data['per_video'])
        rate = f'{data["violation_rate_by_model"][name]:.4f}' if valid else 'N/A (invalid calibration)'
        lines.append(f'| {name} | {n} | {rate} |')
    return '\n'.join(lines)

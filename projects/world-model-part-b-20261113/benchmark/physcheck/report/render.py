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


def render_prompt_fits(summary, records=()):
    lines = ['| Model | Prompt | Videos | Valid fits | g (m/s^2) | T (s) | Mean g relative error | Mean T relative error |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for name, model in sorted(summary['by_model'].items()):
        for prompt, row in sorted(model['per_prompt'].items()):
            errors = {}
            for key in ('g', 'T'):
                values = [r['rel_errs'][key] for r in records if r['model'] == name and r['prompt_id'] == prompt
                          and r.get('fit_status') == 'ok' and key in r.get('rel_errs', {})]
                errors[key] = sum(values) / len(values) if values else None
            values = [row['mean_params'].get('g'), row['mean_params'].get('T'), errors['g'], errors['T']]
            rendered = ['N/A' if v is None else f'{v:.4f}' for v in values]
            lines.append(f'| {name} | {prompt} | {row["n"]} | {row["fit_ok"]}/{row["n"]} | ' + ' | '.join(rendered) + ' |')
    return '\n'.join(lines)


def render_categories(summary):
    valid = summary['valid']
    data = summary if valid else summary['diagnostic_only']
    lines = ['| Model | Category | Valid violation count |', '|---|---|---:|']
    categories = sorted({c for rows in data['violations_by_category'].values() for c in rows})
    for name, rows in sorted(data['violations_by_category'].items()):
        for category in categories:
            value = str(rows.get(category, 0)) if valid else 'N/A (invalid calibration)'
            lines.append(f'| {name} | {category} | {value} |')
    return '\n'.join(lines)

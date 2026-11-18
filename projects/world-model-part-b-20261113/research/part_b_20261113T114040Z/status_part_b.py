"""Summarize retained generation and judgment progress without modifying files."""
import json
from pathlib import Path

run=Path(__file__).resolve().parent/'run'
rows=[]
for model in ['wan2.1_1.3b','cogvideox_5b']:
    path=run/'private'/f'{model}.json'
    if path.exists():
        rows.extend(json.loads(path.read_text()))
judged={p.stem for p in (run/'judgments').glob('*.json')}
print(json.dumps({'generated':len(rows),'production_generated':sum(not r['is_calibration'] for r in rows),
                  'judged':len(judged),'errors':sum(r['status']!='ok' for r in rows),
                  'ready':[r['video_id'] for r in rows if r['status']=='ok' and r['video_id'] not in judged]},indent=2))

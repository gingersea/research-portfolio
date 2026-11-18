"""Verify local copy and complete recomputation from saved model outputs."""
import json
from pathlib import Path
import shutil

import numpy as np

from fetch_tapip import sha

ROOT=Path(__file__).resolve().parent


def main():
    initial=json.loads((ROOT/'handoff_receipt.json').read_text())
    bundle=Path(initial['directory'])
    manifest=json.loads((bundle/'HANDOFF_MANIFEST.json').read_text())
    assert all(sha(bundle/r['file'])==r['sha256'] for r in manifest['files'])
    copied=bundle/'research/observation_gate_20261115'
    recomputed=copied/'handoff_reproduction'
    assert json.loads((recomputed/'receipt.json').read_text())['cases']==40
    for name in ['summary.json','case_metrics.json']:
        assert (ROOT/'comparison'/name).read_bytes()==(recomputed/name).read_bytes()
    for path in (ROOT/'comparison').glob('*_gate.json'):
        assert path.read_bytes()==(recomputed/path.name).read_bytes()
    for path in (ROOT/'comparison').glob('*_masks.npz'):
        first=np.load(path);second=np.load(recomputed/path.name)
        assert first.files==second.files
        assert all(np.array_equal(first[k],second[k]) for k in first.files)
    record={'status':'passed','directory':str(bundle),
            'original_manifest_files_verified':len(manifest['files']),
            'copied_bytes_verified':sum(r['bytes'] for r in manifest['files']),
            'recomputed_sequences':40,'models':['cotracker','tapip3d'],
            'recomputed_scope':'All independent RGB gate decisions/masks and paired center metrics from copied inputs and saved predictions.',
            'summary_bytes_equal':True,'case_metrics_bytes_equal':True,
            'all_gate_json_bytes_equal':True,'all_gate_mask_arrays_equal':True,
            'native_model_inference_rerun_in_handoff':False,
            'native_model_inference_already_verified_in_primary_study':80,
            'portability_limit':'Same workstation and existing isolated ML environment; no cross-machine CUDA/environment recreation claimed.',
            'verification_source_sha256':sha(Path(__file__))}
    target=ROOT/'handoff_reproduction_receipt.json'
    assert not target.exists()
    target.write_text(json.dumps(record,indent=2)+'\n')
    note=ROOT/'REPRODUCTION.zh.md'
    assert not note.exists()
    note.write_text('# 复制后复现核对\n\n'
        '本地交付初始清单的1021个文件逐一通过SHA256核对。随后在交付目录中，'
        '从复制的40段RGB、深度/相机、真值和两模型已保存预测重新计算全部独立观测门及配对中心指标。'
        '汇总、逐序列指标和全部观测门JSON逐字节一致，全部门掩码数组一致。\n\n'
        '复现范围是完整分析重算；没有在交付副本中再次运行模型，也未验证异机环境重建。'
        '原研究已实际完成80次模型推理，检查点与作者源码均保存在交付包。'
        '资产收据保留原机器绝对路径，迁移时需在新研究副本中重新指向assets；'
        '基础PyTorch环境与CUDA工具链不在包内。\n\n'
        '完整结果见REPORT.zh.md；生产科学门仍因身份替换恢复不足而失败。'
        'package_handoff.completed.log是打包完成后的日志，初始包内同名运行中日志为空快照。'
        '所有后加复现材料由REPRODUCTION_MANIFEST.json单独绑定，不覆盖初始清单。\n')
    extras=[(Path(__file__),copied/Path(__file__).name),(target,copied/target.name),
            (note,copied/note.name),(ROOT/'handoff_reproduction.log',copied/'handoff_reproduction.log'),
            (ROOT/'package_handoff.log',copied/'package_handoff.completed.log')]
    for source,destination in extras:
        assert not destination.exists()
        shutil.copy2(source,destination)
        assert sha(source)==sha(destination)
    additions=[]
    for source in [p for p in recomputed.rglob('*') if p.is_file()]+[d for _,d in extras]:
        additions.append({'file':str(source.relative_to(bundle)),'sha256':sha(source),'bytes':source.stat().st_size})
    (bundle/'REPRODUCTION_MANIFEST.json').write_text(json.dumps({'verification':record,'additions':additions},indent=2)+'\n')
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()

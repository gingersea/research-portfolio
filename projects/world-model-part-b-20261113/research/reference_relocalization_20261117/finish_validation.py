"""Continue the frozen comparison once the bounded source prefix is complete."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
sys.path.insert(0,str(PROJECT/'benchmark'))
from physcheck.relations.contracts import save,sha

def record(folder,index,cmd,returncode):
    save(folder/f'{index:02d}.json',{'command':cmd,'returncode':returncode,'orchestrator_sha256':sha(Path(__file__))})

if __name__=='__main__':
    while not(ROOT/'selection.json').exists():
        print(json.dumps({'state':'waiting_for_fixed_source_prefix'}),flush=True)
        time.sleep(30)
    folder=ROOT/'pipeline_steps';folder.mkdir(exist_ok=False)
    py=sys.executable
    def step(index,name,arguments):
        cmd=[py,str(ROOT/name),*arguments]
        with (folder/f'{index:02d}.log').open('x') as f:
            result=subprocess.run(cmd,cwd=PROJECT,stdout=f,stderr=subprocess.STDOUT)
        record(folder,index,cmd,result.returncode)
        assert result.returncode==0,(name,result.returncode)
        print(json.dumps({'done':name}),flush=True)
    step(0,'prepare_validation.py',[])
    cases=json.loads((ROOT/'manifest_validation.json').read_text())['cases']
    if cases:
        native=[py,str(ROOT/'run_native.py'),'--gpu','1']
        descriptor=[py,str(ROOT/'run_descriptor.py'),'--manifest',str(ROOT/'manifest_validation.json'),
            '--freeze',str(ROOT/'validation_freeze.json'),'--output',str(ROOT/'predictions_descriptor_validation'),'--gpu','0']
        with (folder/'01.log').open('x') as f1,(folder/'02.log').open('x') as f2:
            a=subprocess.Popen(native,cwd=PROJECT,stdout=f1,stderr=subprocess.STDOUT)
            b=subprocess.Popen(descriptor,cwd=PROJECT,stdout=f2,stderr=subprocess.STDOUT)
            ra=a.wait();rb=b.wait()
        record(folder,1,native,ra);record(folder,2,descriptor,rb)
        assert ra==rb==0,(ra,rb)
        common=['--manifest',str(ROOT/'manifest_validation.json'),'--native',str(ROOT/'predictions_native_validation'),
                '--descriptor',str(ROOT/'predictions_descriptor_validation')]
        step(3,'analyze.py',common+['--output',str(ROOT/'analysis_validation'),'--split','new_participant_validation'])
        step(4,'pairing_audit.py',common+['--output',str(ROOT/'pairing_validation')])
    step(5,'audit_integrity.py',[])
    step(6,'render_report.py',[])
    print(json.dumps({'bounded_comparison_complete':True,'validation_cases':len(cases)}),flush=True)

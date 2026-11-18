"""Stage-0 action-conditioned system identification and receding-horizon control.

An intentionally easy fully observed linear world. This is a baseline feasibility
run, not a novel visual world model or evidence of real-world physical reasoning.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
STUDY=Path(__file__).resolve().parent
DT=.2
DECAY=.92


def dump(path: Path,value) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def transition(state: np.ndarray,action: np.ndarray) -> np.ndarray:
    velocity=DECAY*state[...,2:]+DT*action
    return np.concatenate([state[...,:2]+DT*velocity,velocity],axis=-1)


def generate(seed: int,count: int,steps: int,diagonal: bool) -> tuple[np.ndarray,np.ndarray]:
    rng=np.random.default_rng(seed)
    choices=np.array([[1,1],[1,-1],[-1,1],[-1,-1]] if diagonal else
                     [[0,0],[1,0],[-1,0],[0,1],[0,-1]],dtype=float)
    actions=choices[rng.integers(len(choices),size=(count,steps))]
    states=np.empty((count,steps+1,4))
    states[:,0,:2]=rng.uniform(-1,1,(count,2))
    states[:,0,2:]=rng.uniform(-.2,.2,(count,2))
    for t in range(steps):
        states[:,t+1]=transition(states[:,t],actions[:,t])
    return states,actions


def fit(states: np.ndarray,actions: np.ndarray,conditioned: bool) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    current=states[:,:-1].reshape(-1,4)
    x=np.concatenate([current,actions.reshape(-1,2),np.ones((len(current),1))],axis=1) if conditioned else np.concatenate([current,np.ones((len(current),1))],axis=1)
    target=states[:,1:].reshape(-1,4)
    coefficients,_,rank,_=np.linalg.lstsq(x,target,rcond=None)
    assert rank==x.shape[1]
    return coefficients[:4].T,coefficients[4:6].T if conditioned else np.zeros((4,2)),coefficients[-1]


def predict(model: tuple,initial: np.ndarray,actions: np.ndarray) -> np.ndarray:
    a,b,c=model
    states=np.empty((len(initial),actions.shape[1]+1,4));states[:,0]=initial
    for t in range(actions.shape[1]):
        states[:,t+1]=states[:,t]@a.T+actions[:,t]@b.T+c
    assert np.isfinite(states).all()
    return states


def planning_gain(model: tuple,horizon: int=12) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    a,b,c=model
    powers=[np.linalg.matrix_power(a,k) for k in range(horizon+1)]
    reach=np.concatenate([powers[horizon-1-k]@b for k in range(horizon)],axis=1)
    q=np.diag([1.,1.,.25,.25])
    gain=np.linalg.solve(reach.T@q@reach+.01*np.eye(2*horizon),reach.T@q)
    offset=sum((power@c for power in powers[:-1]),start=np.zeros(4))
    return gain,powers[-1],offset


def plan(model: tuple,initial: np.ndarray,goal: np.ndarray) -> dict:
    gain,a_h,offset=planning_gain(model)
    states=[initial.copy()];actions=[]
    target=np.r_[goal,[0.,0.]]
    for _ in range(32):
        proposed=gain@(target-a_h@states[-1]-offset)
        action=np.clip(proposed[:2],-1,1)
        assert np.isfinite(action).all() and (np.abs(action)<=1).all()
        actions.append(action);states.append(transition(states[-1],action))
    final_distance=float(np.linalg.norm(states[-1][:2]-goal))
    return {'states':np.asarray(states).tolist(),'actions':np.asarray(actions).tolist(),
            'goal':goal.tolist(),'final_distance':final_distance,'success':final_distance<=.2}


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=STUDY/'results')
    args=parser.parse_args();output=args.output.resolve()
    if output.exists():
        raise ValueError('Choose a new output directory; existing runs are immutable.')
    output.mkdir(parents=True)
    protocol=ROOT/'docs/protocols/action_world_stage0_20261114.md'
    dump(output/'provenance.json',{'started_utc':datetime.now(timezone.utc).isoformat(),
         'source_sha256':digest(Path(__file__)),'protocol_sha256':digest(protocol),
         'python':platform.python_version(),'numpy':np.__version__,
         'dt':DT,'decay':DECAY,'scope':'Fully observed linear synthetic state world; feasibility only.'})
    test_states,test_actions=generate(20261114,128,30,True)
    np.savez(output/'test_data.npz',states=test_states,actions=test_actions)
    rng=np.random.default_rng(20261115)
    starts=np.c_[rng.uniform(-1,1,(64,2)),np.zeros((64,2))]
    angles=rng.uniform(0,2*np.pi,64);radii=rng.uniform(.8,1.5,64)
    goals=starts[:,:2]+radii[:,None]*np.c_[np.cos(angles),np.sin(angles)]
    np.savez(output/'planning_cases.npz',initial=starts,goals=goals)
    oracle=(np.array([[1,0,DT*DECAY,0],[0,1,0,DT*DECAY],[0,0,DECAY,0],[0,0,0,DECAY]]),
            np.array([[DT*DT,0],[0,DT*DT],[DT,0],[0,DT]]),np.zeros(4))
    assert np.allclose(test_states[:,1],test_states[:,0]@oracle[0].T+test_actions[:,0]@oracle[1].T)
    models={};summaries=[];per_trajectory=[];episodes=[]
    for seed in [7,17,27]:
        train_states,train_actions=generate(seed,256,16,False)
        np.savez(output/f'train_seed_{seed}.npz',states=train_states,actions=train_actions)
        for method in ['action_conditioned','action_blind']:
            model=fit(train_states,train_actions,method=='action_conditioned')
            label=f'{method}_seed{seed}';models[label]=model
            np.savez(output/f'{label}.npz',A=model[0],B=model[1],bias=model[2])
            predicted=predict(model,test_states[:,0],test_actions)
            metrics={}
            for h in [1,10,30]:
                delta=predicted[:,h]-test_states[:,h]
                metrics[f'state_rmse_h{h}']=float(np.sqrt(np.mean(delta**2)))
                metrics[f'position_rmse_h{h}']=float(np.sqrt(np.mean(delta[:,:2]**2)))
                metrics[f'velocity_rmse_h{h}']=float(np.sqrt(np.mean(delta[:,2:]**2)))
            summaries.append({'method':method,'train_seed':seed,**metrics})
            for i in range(128):
                per_trajectory.append({'method':method,'train_seed':seed,'trajectory':i,
                    **{f'state_rmse_h{h}':float(np.sqrt(np.mean((predicted[i,h]-test_states[i,h])**2))) for h in [1,10,30]}})
    models['oracle_known_dynamics']=oracle
    planning_summary=[]
    for label,model in models.items():
        runs=[{'episode':i,'model':label,**plan(model,start,goal)} for i,(start,goal) in enumerate(zip(starts,goals))]
        episodes.extend(runs)
        planning_summary.append({'model':label,'episodes':len(runs),'successes':sum(r['success'] for r in runs),
              'median_final_distance':float(np.median([r['final_distance'] for r in runs])),
              'mean_final_distance':float(np.mean([r['final_distance'] for r in runs]))})
    assert len(summaries)==6 and len(per_trajectory)==768 and len(episodes)==448
    dump(output/'prediction_summary.json',summaries)
    dump(output/'prediction_per_trajectory.json',per_trajectory)
    dump(output/'planning_summary.json',planning_summary)
    dump(output/'planning_episodes.json',episodes)
    verification={'passed':True,'learned_models':6,'oracle_models':1,'train_trajectories_per_seed':256,
                  'train_transitions_per_seed':4096,'heldout_test_trajectories':128,
                  'paired_planning_cases':64,'episode_runs':448,'bounds_verified':True,
                  'all_results_retained':True,'visual_model_trained':False}
    dump(output/'verification.json',verification)
    lines=['# Action-conditioned world model: stage-0 feasibility','',
           'Fully observed, noiseless linear dynamics; least-squares baselines. This is intentionally easy and not a novel visual world model.',
           'New diagonal action combinations are held out from training; this does not establish nonlinear or visual OOD generalization.', '',
           '| Method | Data seed | Position RMSE H10 | Position RMSE H30 |',
           '|---|---:|---:|---:|']
    for row in summaries:
        lines.append(f'| {row["method"]} | {row["train_seed"]} | {row["position_rmse_h10"]:.6g} | {row["position_rmse_h30"]:.6g} |')
    lines+=['','| Planning model | Successes / same64 cases | Median final distance |','|---|---:|---:|']
    for row in planning_summary:
        lines.append(f'| {row["model"]} | {row["successes"]}/{row["episodes"]} | {row["median_final_distance"]:.6g} |')
    lines+=['','The64 cases are reused across models;448 runs are not448 independent cases. Success means final distance<=0.2 after32 controls.',
            'The action-blind planner has B=0 by construction. This demonstrates the need for an action interface, not superiority over competing modern world models.',
            'Oracle uses known dynamics and is not learned. No P values or real-task population confidence intervals are reported.',
            'Only state/action system identification and closed-loop wiring are demonstrated. Pixels, occlusion, contact, partial observability and changed dynamics are untested.',
            'Next step is a separately designed pixel/latent model study with multiple dynamical worlds, matched compute and episode-level splits; retain oracle/blind controls.',
            '', 'Reproduce with `python research/action_world_stage0_20261114/experiment.py --output NEW_DIRECTORY`.']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    dump(output/'checksums.json',{p.relative_to(output).as_posix():digest(p)
                                for p in sorted(output.rglob('*')) if p.is_file()})
    print(json.dumps({'verification':verification,'prediction':summaries,'planning':planning_summary},indent=2))


if __name__=='__main__':main()

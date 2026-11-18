"""Matched-budget synthetic-only multiscale segmentation experiment.

No generated-video ground truth is inferred. Checkpoints selected on validation
are frozen before the new test set is rendered. All outputs use a new directory.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO/'benchmark'))
from training_pilot import TinyUNet, make_dataset, segmentation_loss


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')


def digest_arrays(images, masks):
    return hashlib.sha256(images.tobytes()+masks.tobytes()).hexdigest()


def render(n_positive, n_negative, seed):
    x, y, _ = make_dataset(n_positive, seed)
    positives=(x.numpy().transpose(0,2,3,1)*255).round().astype(np.uint8)
    masks=y.numpy()[:,0].astype(np.uint8)
    rng=np.random.default_rng(seed+10)
    negatives=[]
    for _ in range(n_negative):
        im=np.clip(rng.uniform(180,255)+rng.normal(0,3,(96,96,3)),0,255).astype(np.uint8)
        for _ in range(int(rng.integers(1,4))):
            endpoints=rng.integers(0,96,size=(2,2))
            # Arrays are RGB; OpenCV writes the specified tuple without conversion.
            cv2.line(im,tuple(endpoints[0].tolist()),tuple(endpoints[1].tolist()),(170,30,30),int(rng.integers(1,3)))
        negatives.append(im)
    images=np.concatenate([positives,np.stack(negatives)])
    truth=np.concatenate([masks,np.zeros((n_negative,96,96),dtype=np.uint8)])
    assert np.all(truth[:n_positive].sum((1,2))>0)
    assert not truth[n_positive:].any()
    return images,truth


def augment(images, truth, seed):
    rng=np.random.default_rng(seed)
    result=np.full_like(images,255)
    masks=np.zeros_like(truth)
    for i,(im,mask) in enumerate(zip(images,truth)):
        size=int(rng.integers(16,97))
        x,y=rng.integers(0,97-size,size=2)
        result[i,y:y+size,x:x+size]=cv2.resize(im,(size,size),interpolation=cv2.INTER_AREA)
        masks[i,y:y+size,x:x+size]=cv2.resize(mask,(size,size),interpolation=cv2.INTER_NEAREST)
    assert np.array_equal(masks.any((1,2)),truth.any((1,2)))
    return result,masks


def tensors(images, masks, device):
    return (torch.from_numpy(images.transpose(0,3,1,2).copy()).float().to(device)/255,
            torch.from_numpy(masks[:,None].copy()).float().to(device))


@torch.no_grad()
def predict(model, images, device):
    model.eval()
    x=torch.from_numpy(images.transpose(0,3,1,2).copy()).float()/255
    return torch.cat([model(b.to(device)).sigmoid()[:,0].cpu() for b in x.split(32)]).numpy()>.5


def frame_metrics(pred, truth):
    positive=bool(truth.any())
    if not positive:
        return {'positive':False,'iou':None,'centroid_error_px':None,'nonempty':bool(pred.any())}
    intersection=int((pred&truth).sum());union=int((pred|truth).sum())
    py,px=np.nonzero(pred);ty,tx=np.nonzero(truth)
    return {'positive':True,'iou':intersection/union,
            'centroid_error_px':float(np.hypot(px.mean()-tx.mean(),py.mean()-ty.mean())) if len(px) else float(np.hypot(*truth.shape)),
            'nonempty':bool(len(px))}


def validation_score(pred,truth):
    positive=truth.any((1,2))
    inter=(pred&truth).sum((1,2));union=(pred|truth).sum((1,2))
    iou=float(np.mean(inter[positive]/union[positive]))
    fpr=float(pred[~positive].any((1,2)).mean())
    return {'positive_iou':iou,'negative_fpr':fpr,'selection_score':.8*iou+.2*(1-fpr)}


def train(output,device):
    train_im,train_y=render(1536,512,61001)
    valid_im,valid_y=augment(*render(256,64,62001),62101)
    augmented,augmented_y=augment(train_im,train_y,61101)
    hashes={'train_single':digest_arrays(train_im,train_y),'train_multiscale':digest_arrays(augmented,augmented_y),
            'validation':digest_arrays(valid_im,valid_y)}
    dump(output/'development_hashes.json',hashes)
    # Same validation images and scoring, budget and optimizer for both arms.
    for arm,images,labels in [('single_scale',train_im,train_y),('multiscale',augmented,augmented_y)]:
        x,y=tensors(images,labels,device)
        for seed in [7,17,27]:
            torch.manual_seed(seed)
            model=TinyUNet().to(device)
            optimizer=torch.optim.Adam(model.parameters(),lr=.001)
            dest=output/'models'/arm/f'seed_{seed}'
            dest.mkdir(parents=True)
            history=[];best=-float('inf');selected=None
            start=time.monotonic()
            for epoch in range(1,13):
                model.train();losses=[]
                for indices in torch.randperm(2048,device=device).split(64):
                    optimizer.zero_grad(set_to_none=True)
                    loss=segmentation_loss(model(x[indices]),y[indices])
                    loss.backward();optimizer.step();losses.append(float(loss.item()))
                score=validation_score(predict(model,valid_im,device),valid_y.astype(bool))
                record={'epoch':epoch,'train_loss':float(np.mean(losses)),**score}
                history.append(record)
                if score['selection_score']>best:
                    best=score['selection_score'];selected=epoch
                    torch.save(model.state_dict(),dest/'best.pt')
            dump(dest/'training.json',{'arm':arm,'seed':seed,'epochs':12,'batch_size':64,'lr':.001,
                    'loss':'BCEWithLogits + softDice','selected_epoch':selected,'history':history,
                    'checkpoint_sha256':sha(dest/'best.pt'),'runtime_seconds':time.monotonic()-start})
            print(f'{arm} seed {seed}: selected epoch {selected}, validation score {best:.4f}',flush=True)
    dump(output/'checkpoint_freeze.json',{
        str(path.relative_to(output)):sha(path) for path in sorted((output/'models').rglob('best.pt'))})


def evaluate(output,dest,device,old_checkpoint):
    dest.mkdir(parents=True,exist_ok=False)
    # Explicitly called only after every trained checkpoint has been frozen.
    frozen=json.loads((output/'checkpoint_freeze.json').read_text())
    assert len(frozen)==6
    for name,digest in frozen.items():assert sha(output/name)==digest
    originals,masks=render(256,64,63001)
    assert np.array_equal(masks.any((1,2)),np.arange(320)<256)
    dump(dest/'test_design.json',{'seed':63001,'n_positive':256,'n_negative':64,
                                'source_tensor_sha256':digest_arrays(originals,masks),
                                'checkpoint_freeze_sha256':sha(output/'checkpoint_freeze.json')})
    reference=json.loads(old_checkpoint.with_name('metrics.json').read_text())
    assert sha(old_checkpoint)==reference['model_sha256']
    methods={'legacy':old_checkpoint}
    for arm in ['single_scale','multiscale']:
        for seed in [7,17,27]:methods[f'{arm}_{seed}']=output/'models'/arm/f'seed_{seed}'/'best.pt'
    models={}
    for name,path in methods.items():
        model=TinyUNet().to(device).eval()
        model.load_state_dict(torch.load(path,map_location=device,weights_only=True))
        models[name]=model
    records=[];dataset_hashes={}
    for size in [24,48,96]:
        canvas=np.full((320,112,192,3),255,dtype=np.uint8)
        truth=np.zeros((320,112,192),dtype=np.uint8)
        cy,cx=(112-size)//2,(192-size)//2
        for i,(im,mask) in enumerate(zip(originals,masks)):
            canvas[i,cy:cy+size,cx:cx+size]=cv2.resize(im,(size,size),interpolation=cv2.INTER_AREA)
            truth[i,cy:cy+size,cx:cx+size]=cv2.resize(mask,(size,size),interpolation=cv2.INTER_NEAREST)
        truth=truth.astype(bool)
        assert np.array_equal(truth.any((1,2)),np.arange(320)<256)
        for strategy in ['native','warp96','letterbox96']:
            if strategy=='native':im=canvas
            elif strategy=='warp96':im=np.stack([cv2.resize(v,(96,96),interpolation=cv2.INTER_AREA) for v in canvas])
            else:
                im=np.full((320,96,96,3),255,dtype=np.uint8)
                im[:,20:76]=np.stack([cv2.resize(v,(96,56),interpolation=cv2.INTER_AREA) for v in canvas])
            dataset_hashes[f'{size}_{strategy}']=digest_arrays(im,truth)
            for method in [*models,'color']:
                if method=='color':
                    red,green,blue=im.astype(np.float32).transpose(3,0,1,2)
                    predicted=(red>80)&(red>1.4*green)&(red>1.4*blue)
                else:predicted=predict(models[method],im,device)
                for i,(pm,tm) in enumerate(zip(predicted,truth)):
                    if strategy!='native':
                        if strategy=='letterbox96':pm=pm[20:76]
                        pm=cv2.resize(pm.astype(np.uint8),(192,112),interpolation=cv2.INTER_NEAREST).astype(bool)
                    records.append({'scene_id':i,'patch_size':size,'strategy':strategy,'method':method,**frame_metrics(pm,tm)})
            print(f'Test patch {size}, {strategy}: 320 scenes, 8 methods',flush=True)
    dump(dest/'per_scene.json',records)
    dump(dest/'input_hashes.json',dataset_hashes)
    assert len(records)==320*9*8
    summary=[]
    for size in [24,48,96]:
        for strategy in ['native','warp96','letterbox96']:
            for method in [*models,'color']:
                subset=[r for r in records if r['patch_size']==size and r['strategy']==strategy and r['method']==method]
                positive=[r for r in subset if r['positive']];negative=[r for r in subset if not r['positive']]
                assert len(positive)==256 and len(negative)==64
                summary.append({'patch_size':size,'strategy':strategy,'method':method,'n_positive':256,'n_negative':64,
                            'positive_iou':float(np.mean([r['iou'] for r in positive])),
                            'positive_centroid_error_px':float(np.mean([r['centroid_error_px'] for r in positive])),
                            'positive_detection_rate':float(np.mean([r['nonempty'] for r in positive])),
                            'negative_fpr':float(np.mean([r['nonempty'] for r in negative]))})
    dump(dest/'summary.json',summary)
    rng=np.random.default_rng(20261111)
    bootstrap=rng.integers(0,256,size=(2000,256))
    effects=[]
    for size in [24,48,96]:
        for strategy in ['native','warp96','letterbox96']:
            per_seed=[]
            for seed in [7,17,27]:
                a=np.array([r['iou'] for r in records if r['patch_size']==size and r['strategy']==strategy and r['method']==f'multiscale_{seed}' and r['positive']])
                b=np.array([r['iou'] for r in records if r['patch_size']==size and r['strategy']==strategy and r['method']==f'single_scale_{seed}' and r['positive']])
                per_seed.append(a-b)
            differences=np.mean(per_seed,axis=0)
            effects.append({'patch_size':size,'strategy':strategy,
                'primary':size==24 and strategy=='warp96',
                'mean_iou_difference':float(differences.mean()),
                'ci95_conditional_on_three_training_seeds':np.quantile(differences[bootstrap].mean(1),[.025,.975]).tolist(),
                'per_seed_mean_differences':np.mean(per_seed,axis=1).tolist(),
                'n_paired_scenes':256,'n_training_seeds':3})
    dump(dest/'effects.json',effects)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--old-checkpoint',type=Path,required=True)
    parser.add_argument('--eval-only',action='store_true')
    parser.add_argument('--evaluation-name',default='test')
    args=parser.parse_args()
    torch.set_num_threads(8)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    device='cuda:0' if torch.cuda.is_available() else 'cpu'
    if not args.eval_only:
        args.output.mkdir(parents=True,exist_ok=False)
        train(args.output,device)
        dump(args.output/'provenance.json',{'script_sha256':sha(__file__),
                    'training_source_sha256':sha(REPO/'benchmark/training_pilot.py'),
                    'old_checkpoint_sha256':sha(args.old_checkpoint),
                    'python':sys.version,'device':device,'torch_runtime':str(torch.__version__),
                    'versions':{p:importlib.metadata.version(p) for p in ['torch','numpy','opencv-python-headless']},
                    'train_seeds':[7,17,27],'test_seed':63001,'render_seeds':{'train':61001,'validation':62001},
                    'fixed_epochs':12,'fixed_threshold':.5,'scope':'synthetic only; no generator fine-tuning or human labels'})
    else:
        assert sha(__file__)==json.loads((args.output/'provenance.json').read_text())['script_sha256']
    evaluate(args.output,args.output/args.evaluation_name,device,args.old_checkpoint)


if __name__=='__main__':main()

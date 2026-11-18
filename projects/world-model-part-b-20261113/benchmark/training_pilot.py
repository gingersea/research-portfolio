"""Supervised ball segmentation pilot for the course milestone.

This separate experiment does not replace or re-score the frozen Part A audit.
All train/validation/test images are procedurally rendered, not model videos.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def make_dataset(n: int, seed: int, size: int = 96):
    rng = np.random.default_rng(seed)
    images, masks, centers = [], [], []
    yy, xx = np.mgrid[:size, :size]
    for _ in range(n):
        radius = int(rng.integers(max(2, round(size*.06)), max(4, round(size*.14))))
        cx, cy = rng.integers(radius+3, size-radius-3, size=2).tolist()
        bg = rng.uniform(180, 255) + rng.uniform(-25, 25)*xx/size
        image = np.repeat(bg[..., None], 3, axis=2).astype(np.float32)
        if rng.random() < .5:
            cv2.ellipse(image, (cx+radius, cy+radius), (radius+3, max(2,radius//3)),
                        0, 0, 360, (150,150,150), -1)
        if rng.random() < .4:
            cv2.line(image, (cx,0), (cx,cy), (170,30,30), int(rng.integers(1,3)))
        mask = ((xx-cx)**2 + (yy-cy)**2 <= radius**2)
        color = np.array([rng.uniform(130,250), rng.uniform(5,65), rng.uniform(5,55)])
        lighting = np.clip(1 - .3*(xx-cx)/radius, .6, 1.3)
        image[mask] = (lighting[...,None]*color)[mask]
        if rng.random() < .5:
            image = cv2.GaussianBlur(image, (5,5), float(rng.uniform(.4,1.3)))
        image += rng.normal(0, 3, image.shape)
        images.append(np.clip(image,0,255).astype(np.uint8))
        masks.append(mask.astype(np.float32))
        centers.append((cx,cy))
    x = torch.from_numpy(np.stack(images).transpose(0,3,1,2).copy()).float()/255
    y = torch.from_numpy(np.stack(masks)[:,None])
    return x, y, centers


class TinyUNet(nn.Module):
    def __init__(self):
        super().__init__()
        def block(cin, cout):
            return nn.Sequential(nn.Conv2d(cin,cout,3,padding=1), nn.ReLU(),
                                 nn.Conv2d(cout,cout,3,padding=1), nn.ReLU())
        self.enc1 = block(3,8)
        self.enc2 = block(8,16)
        self.bottleneck = block(16,32)
        self.up2 = nn.ConvTranspose2d(32,16,2,stride=2)
        self.dec2 = block(32,16)
        self.up1 = nn.ConvTranspose2d(16,8,2,stride=2)
        self.dec1 = block(16,8)
        self.out = nn.Conv2d(8,1,1)

    def forward(self, x):
        a = self.enc1(x)
        b = self.enc2(F.max_pool2d(a,2))
        c = self.bottleneck(F.max_pool2d(b,2))
        d = self.dec2(torch.cat([self.up2(c),b],dim=1))
        return self.out(self.dec1(torch.cat([self.up1(d),a],dim=1)))


def segmentation_loss(logits, labels):
    p = logits.sigmoid()
    dice = 1 - (2*(p*labels).sum((1,2,3))+1)/(p.sum((1,2,3))+labels.sum((1,2,3))+1)
    return F.binary_cross_entropy_with_logits(logits,labels) + dice.mean()


def mask_metrics(predicted, labels):
    pred = predicted.detach().cpu().numpy()[:,0] > .5
    truth = labels.detach().cpu().numpy()[:,0] > .5
    inter = (pred & truth).sum((1,2))
    union = (pred | truth).sum((1,2))
    errors = []
    for pm, tm in zip(pred,truth):
        py,px = np.nonzero(pm)
        ty,tx = np.nonzero(tm)
        errors.append(float(np.hypot(px.mean()-tx.mean(),py.mean()-ty.mean()))
                      if len(px) else float(np.hypot(*pm.shape)))
    return {"mean_iou":float(np.mean(inter/np.maximum(union,1))),
            "centroid_error_px":float(np.mean(errors)),
            "detection_rate":float(np.mean(pred.sum((1,2))>0)), "n":len(pred)}


@torch.no_grad()
def evaluate(model, x, y, batch_size=64):
    model.eval()
    preds, losses = [], []
    for start in range(0,len(x),batch_size):
        logits = model(x[start:start+batch_size])
        preds.append(logits.sigmoid().cpu())
        losses.append((len(logits),segmentation_loss(logits,y[start:start+batch_size]).item()))
    return {**mask_metrics(torch.cat(preds),y),
            "loss":sum(n*v for n,v in losses)/len(x)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=Path("benchmark/results/training_pilot"))
    parser.add_argument("--epochs",type=int,default=12)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(8)
    torch.manual_seed(7)
    np.random.seed(7)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    config = {"model":"TinyUNet", "channels":[8,16,32], "seed":7, "image_size":96,
              "n_train":2048,"n_validation":256,"n_test":256,
              "split_seeds":{"train":10001,"validation":20001,"test":30001},
              "epochs":args.epochs,"batch_size":64,"optimizer":"Adam","lr":.001,
              "loss":"BCEWithLogits + soft Dice", "mask_threshold":.5,
              "selection":"maximum validation mean IoU", "device":device}
    data, hashes = {}, {}
    for split,n in [("train",2048),("validation",256),("test",256)]:
        x,y,_ = make_dataset(n,config["split_seeds"][split])
        digest = hashlib.sha256(x.numpy().tobytes()+y.numpy().tobytes()).hexdigest()
        hashes[split] = digest
        data[split] = (x.to(device),y.to(device))
    model = TinyUNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=.001)
    initial = evaluate(model,*data["validation"])
    history, best, best_epoch = [], -1, None
    start_time = time.monotonic()
    for epoch in range(1,args.epochs+1):
        model.train()
        permutation = torch.randperm(2048,device=device)
        losses = []
        for indices in permutation.split(64):
            x,y = (d[indices] for d in data["train"])
            optimizer.zero_grad(set_to_none=True)
            loss = segmentation_loss(model(x),y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        validation = evaluate(model,*data["validation"])
        row = {"epoch":epoch,"train_loss":float(np.mean(losses)),"validation":validation}
        history.append(row)
        print(json.dumps(row),flush=True)
        if validation["mean_iou"] > best:
            best,best_epoch = validation["mean_iou"],epoch
            torch.save(model.state_dict(),args.output/"best_model.pt")
    model.load_state_dict(torch.load(args.output/"best_model.pt",weights_only=True,map_location=device))
    test = evaluate(model,*data["test"])
    x,y = data["test"]
    red = x[:,0]
    baseline = ((red > 80/255) & (red > 1.4*x[:,1]) & (red > 1.4*x[:,2])).float()[:,None]
    result = {"config":config,"parameter_count":sum(p.numel() for p in model.parameters()),
              "initial_validation":initial,"history":history,"selected_epoch":best_epoch,
              "test":test,"color_threshold_test":mask_metrics(baseline,y),
              "runtime_seconds":time.monotonic()-start_time,
              "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "dataset_hashes":hashes,"torch_version":torch.__version__,
              "model_sha256":hashlib.sha256((args.output/"best_model.pt").read_bytes()).hexdigest(),
              "scope":"synthetic-only supervised pilot; not applied to the frozen 24-video audit"}
    (args.output/"metrics.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ("history","dataset_hashes")},indent=2))


if __name__ == "__main__":
    main()

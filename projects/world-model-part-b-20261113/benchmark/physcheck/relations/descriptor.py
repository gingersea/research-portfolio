"""Global cosine correspondence; reciprocity is not a visibility estimate."""
import torch
import torch.nn.functional as F


def descriptor_match(reference, target, query_xy):
    """Match HxWxD descriptors; return forward/back XY and cycle acceptance.

    Bilinear sampling uses image pixel centers (align_corners=True) and border
    clamping, as in the released PIPs sampler. Invalid initial queries and zero
    descriptors yield no candidate. Argmax ties use the first row-major pixel.
    """
    if reference.ndim!=3 or target.shape!=reference.shape or query_xy.ndim!=2 or query_xy.shape[1]!=2:
        raise ValueError('expected equal HxWxD maps and Nx2 query coordinates')
    if not torch.isfinite(reference).all() or not torch.isfinite(target).all():
        raise ValueError('descriptor maps must be finite')
    h,w,d=reference.shape
    q=query_xy.to(device=reference.device,dtype=reference.dtype)
    valid=torch.isfinite(q).all(-1)&(q[:,0]>=0)&(q[:,0]<w)&(q[:,1]>=0)&(q[:,1]<h)
    safe=torch.where(valid[:,None],q,torch.zeros_like(q))
    grid=torch.stack((safe[:,0]*2/max(w-1,1)-1,safe[:,1]*2/max(h-1,1)-1),-1)[None,None]
    sampled=F.grid_sample(reference.permute(2,0,1)[None],grid,align_corners=True,padding_mode='border')[0,:,0].T
    valid &= sampled.norm(dim=-1)>1e-8
    ref=F.normalize(reference.reshape(-1,d),dim=-1,eps=1e-8)
    tar=F.normalize(target.reshape(-1,d),dim=-1,eps=1e-8)
    target_valid=target.reshape(-1,d).norm(dim=-1)>1e-8
    reference_valid=reference.reshape(-1,d).norm(dim=-1)>1e-8
    similarity=F.normalize(sampled,dim=-1,eps=1e-8)@tar.T
    similarity[:,~target_valid]=-torch.inf
    index=similarity.argmax(-1)
    valid &= torch.isfinite(similarity.max(-1).values)
    xy=torch.stack((index%w,index//w),-1).to(q.dtype)
    reverse_sim=tar[index]@ref.T
    reverse_sim[:,~reference_valid]=-torch.inf
    reverse_index=reverse_sim.argmax(-1)
    back=torch.stack((reverse_index%w,reverse_index//w),-1).to(q.dtype)
    distance=((back-q)*q.new_tensor([256/w,256/h])).norm(dim=-1)
    accepted=valid&torch.isfinite(reverse_sim.max(-1).values)&(distance<4)
    xy[~valid]=torch.nan
    back[~valid]=torch.nan
    return xy,back,accepted

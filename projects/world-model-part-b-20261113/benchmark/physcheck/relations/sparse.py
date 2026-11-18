"""Sparse-label point measurement: unknown frames are not negative references."""
import numpy as np


def inside(xy, shape):
    height,width=shape
    return np.isfinite(xy).all(axis=-1) & (xy[...,0]>=0) & (xy[...,0]<width) & (xy[...,1]>=0) & (xy[...,1]<height)


def reciprocal_accept(candidate, native, reverse, reverse_visible, reference, shape):
    height,width=shape
    error=np.linalg.norm((reverse-reference)*np.array([256/width,256/height]),axis=-1)
    return native & inside(candidate,shape) & reverse_visible & inside(reverse,shape) & (error<4)


def sparse_counts(xy, declared, truth, shape):
    known=truth['visibility_valid'].astype(bool).copy()
    known[0]=False
    visible=truth['visible'].astype(bool)
    coordinate_valid=truth['coordinate_valid'].astype(bool)
    out=truth['out_of_view'].astype(bool)
    positive=known & visible & coordinate_valid
    negative=known & ~visible
    if xy is None:
        xy=np.full_like(truth['xy'],np.nan,dtype=float)
        declared=np.zeros_like(known)
    height,width=shape
    error=np.linalg.norm((xy-truth['xy'])*np.array([256/width,256/height]),axis=-1)
    located=np.isfinite(error) & (error<4)
    accepted=known & declared
    correct=accepted & positive & located
    returned=recovered=strict=0
    for point in range(known.shape[1]):
        pending=False
        wrong_oov=False
        for frame in range(1,len(known)):
            if known[frame,point] and out[frame,point]:
                pending=True
                wrong_oov|=bool(declared[frame,point])
            elif pending and positive[frame,point]:
                returned+=1
                recovered+=int(correct[frame,point])
                strict+=int(correct[frame,point] and not wrong_oov)
                pending=wrong_oov=False
    return {'positive':int(positive.sum()), 'negative':int(negative.sum()),
            'accepted':int(accepted.sum()), 'correct':int(correct.sum()),
            'false_visible':int((accepted & negative).sum()),
            'false_visible_oov':int((accepted & negative & out).sum()),
            'negative_oov':int((negative & out).sum()),
            'false_visible_occluded':int((accepted & negative & ~out).sum()),
            'negative_occluded':int((negative & ~out).sum()),
            'wrong_location':int((accepted & positive & ~located).sum()),
            'unverifiable_accept':int((accepted & visible & ~coordinate_valid).sum()),
            'located_all_visible':int((positive & located).sum()),
            'returned_points':returned, 'recovered':recovered, 'strict_recovered':strict}

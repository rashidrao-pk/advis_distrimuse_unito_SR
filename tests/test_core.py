import numpy as np
import torch
from r2ad.core import boxes,random_boxes,extract,Encoder,Decoder,Discriminator,infer_map

def test_boxes():
    m=np.ones((360,360),dtype=np.uint8)
    bb=boxes(m,224,64,.85)
    assert len(bb)==16
    assert all(0<=x<=136 and 0<=y<=136 for x,y,_,_ in bb)

def test_random():
    m=np.ones((360,360),dtype=np.uint8)
    rr=random_boxes(m,224,6,np.random.default_rng(1))
    assert len(rr)==6

def test_architecture():
    e=Encoder(64);d=Decoder(64);x=torch.zeros(1,3,128,128)
    with torch.inference_mode():
        mu,lv=e(x);y=d(mu)
    assert mu.shape==(1,64) and y.shape==x.shape

"""Atomic training snapshots, resumable batch order, and graceful pause signals."""
import random
import signal
from pathlib import Path

import numpy as np
import torch


def atomic_save(value,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    torch.save(value,temporary)
    temporary.replace(path)


def capture_rng():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'].cpu())
    if state.get('cuda') is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


class ResumableBatches:
    def __init__(self,size,batch_size,seed):
        self.size,self.batch_size,self.seed=size,batch_size,seed
        self.order=None;self.start_batch=0

    def set_epoch(self,epoch,order=None,start_batch=0):
        generator=torch.Generator().manual_seed(self.seed+epoch)
        self.order=order if order is not None else torch.randperm(self.size,generator=generator).tolist()
        if len(self.order)!=self.size or sorted(self.order)!=list(range(self.size)):
            raise ValueError('Saved training order does not match dataset')
        self.start_batch=start_batch

    def __iter__(self):
        for start in range(self.start_batch*self.batch_size,self.size,self.batch_size):
            yield self.order[start:start+self.batch_size]

    def __len__(self):
        return (self.size+self.batch_size-1)//self.batch_size-self.start_batch


class PauseSignals:
    """First signal requests a pause; a second interrupt aborts immediately."""
    def __init__(self):
        self.requested=False;self.previous={}

    def request(self,signum=None,frame=None):
        if self.requested:
            raise KeyboardInterrupt
        self.requested=True
        print('\n[pause] Requested; saving after the current safe training boundary. Please wait.',flush=True)

    def __enter__(self):
        for name in ('SIGINT','SIGTERM','SIGUSR1'):
            signum=getattr(signal,name,None)
            if signum is not None:
                self.previous[signum]=signal.signal(signum,self.request)
        return self

    def __exit__(self,*args):
        for signum,handler in self.previous.items():
            signal.signal(signum,handler)

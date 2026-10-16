#!/usr/bin/env python3
"""Stress-test R2AD on NORMAL held-out frames. Reports false positive rate only."""
import argparse,csv,json,sys
from pathlib import Path
import cv2,numpy as np,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from r2ad.core import images,read_mask,load_models,infer_map,transform_frame

def main():
    p=argparse.ArgumentParser()
    for a in ('input','mask','checkpoint','threshold-json','output-dir'):p.add_argument('--'+a,required=True)
    p.add_argument('--max-frames',type=int,default=30);p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--device',default='auto');p.add_argument('--translate',nargs='+',type=float,default=[0,5,10,20,30])
    p.add_argument('--rotate',nargs='+',type=float,default=[0,1,2,3])
    p.add_argument('--scale',nargs='+',type=float,default=[1,.95,1.05])
    args=p.parse_args();cfg=json.loads(Path(args.threshold_json).read_text())
    device=('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
    enc,dec,_=load_models(args.checkpoint,device)
    paths=images(args.input)[:args.max_frames];out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    experiments=[('translation_x',v,dict(tx=v)) for v in args.translate]+[('rotation',v,dict(angle=v)) for v in args.rotate]+[('scale',v,dict(scale=v)) for v in args.scale]
    records=[]
    for kind,value,tform in experiments:
        scores=[];covered=[]
        for path in paths:
            img=cv2.imread(str(path))
            if img is None:raise RuntimeError(f'Cannot read {path}')
            mask=read_mask(args.mask,img.shape)
            moved=transform_frame(img,**tform)
            r=infer_map(moved,mask,enc,dec,device,cfg['patch_size'],cfg['stride'],cfg['min_coverage'],args.batch_size,cfg['sigma'])
            scores.append(r['score']);covered.append(r['coverage_ratio'])
        scores=np.asarray(scores)
        row={'perturbation':kind,'value':value,'n':len(scores),'fpr':float(np.mean(scores>cfg['threshold'])),
             'mean_score':float(scores.mean()),'std_score':float(scores.std()),'coverage_ratio':float(np.mean(covered))}
        records.append(row);print(row)
    with (out/'robustness.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for kind in sorted(set(r['perturbation'] for r in records)):
        data=sorted((r for r in records if r['perturbation']==kind),key=lambda r:r['value'])
        plt.figure(figsize=(7,4));plt.plot([r['value'] for r in data],[r['fpr'] for r in data],marker='o')
        plt.xlabel(kind.replace('_',' '));plt.ylabel('False positive rate (normal frames)')
        plt.ylim(-.03,1.03);plt.grid(alpha=.3);plt.tight_layout()
        plt.savefig(out/f'fpr_{kind}.png',dpi=220);plt.close()
if __name__=='__main__':main()

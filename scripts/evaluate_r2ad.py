#!/usr/bin/env python3
import argparse,csv,json,sys
from pathlib import Path
import cv2,numpy as np,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from r2ad.core import images,read_mask,load_models,infer_map,save_map,transform_frame
from r2ad.config import resolve, DEFAULT_CONFIG, load_config
from r2ad.mcap_data import load_manifest, McapFrameReader

def main():
    p=argparse.ArgumentParser(description='R2AD inference, validation threshold calibration and shift tests')
    p.add_argument('--dataset-source',choices=('images','mcap'),default='images')
    p.add_argument('--manifest',help='Indexed MCAP manifest')
    p.add_argument('--split',choices=('train','val','test'),help='MCAP split override')
    p.add_argument('--config',default=str(DEFAULT_CONFIG));p.add_argument('--safety-area',required=True,choices=('PLeft','PRight','RoboArm','ConvBelt','SafetyZone'))
    p.add_argument('--input');p.add_argument('--mask');p.add_argument('--checkpoint')
    p.add_argument('--output-dir');p.add_argument('--threshold-json')
    p.add_argument('--calibrate',action='store_true',help='Compute a normal-only percentile threshold')
    p.add_argument('--threshold-percentile',type=float,default=99.0)
    p.add_argument('--patch-size',type=int,default=224);p.add_argument('--stride',type=int,default=64)
    p.add_argument('--min-coverage',type=float,default=.85);p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--sigma',type=float,default=1.);p.add_argument('--save-maps',action='store_true')
    p.add_argument('--tx',type=float,default=0);p.add_argument('--ty',type=float,default=0)
    p.add_argument('--angle',type=float,default=0);p.add_argument('--scale',type=float,default=1)
    p.add_argument('--device',default='auto');p.add_argument('--max-frames',type=int,default=0)
    args=p.parse_args()
    if args.dataset_source=='mcap':
        cfg=load_config(args.config);root=Path(cfg['data']['dataset_base']).expanduser();rcfg=cfg.get('r2ad') or {}
        mask_cfg=(rcfg.get('safety_zone_mask') or cfg.get('data',{}).get('mask_types',{}).get('SafetyZone')) if args.safety_area=='SafetyZone' else cfg['data']['mask_types'].get(args.safety_area)
        args.mask=args.mask or mask_cfg
        if not args.mask:raise ValueError('No area mask configured')
        args.mask=str(root/args.mask if not Path(args.mask).is_absolute() else args.mask)
        args.manifest=args.manifest or rcfg.get('manifest') or str(root/'r2ad'/'manifest.csv')
        args.manifest=str(root/args.manifest if not Path(args.manifest).is_absolute() else args.manifest)
        base=Path(cfg.get('results',{}).get('output_dir','results'))/str(rcfg.get('dataset_version','V6'))/'r2ad'/args.safety_area
        args.checkpoint=args.checkpoint or str(base/'model_r2ad.pt')
        args.output_dir=args.output_dir or str(base/('calibration' if args.calibrate else 'test'))
        if not args.threshold_json and not args.calibrate:args.threshold_json=str(base/'calibration'/'threshold.json')
        args.split=args.split or ('val' if args.calibrate else 'test')
        if args.calibrate and args.split!='val':p.error('Calibration must use normal validation split')
    else:args=resolve(args,'calibrate' if args.calibrate else 'evaluate')
    device=('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
    enc,dec,ck=load_models(args.checkpoint,device)
    if args.dataset_source=='mcap':
        paths=load_manifest(args.manifest,args.split)
        if args.calibrate and any(r['label']!='normal' for r in paths):raise ValueError('Calibration must be normal-only')
        reader=McapFrameReader()
    else:paths=images(args.input)
    if args.max_frames>0:paths=paths[:args.max_frames]
    if not paths:raise ValueError('No input frames')
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    if args.calibrate and (args.tx or args.ty or args.angle or args.scale!=1.):
        p.error('Calibrate on unshifted normal validation images only')
    tau=None
    if args.threshold_json and not args.calibrate:
        cfg=json.loads(Path(args.threshold_json).read_text())
        expected={k:getattr(args,k) for k in ('patch_size','stride','min_coverage','sigma')}
        for k,v in expected.items():
            if cfg.get(k)!=v:raise ValueError(f'Threshold config mismatch for {k}: calibrated {cfg.get(k)}, current {v}')
        tau=float(cfg['threshold'])
    records=[]
    for i,path in enumerate(paths):
        frame=reader.read(path) if args.dataset_source=='mcap' else cv2.imread(str(path))
        if frame is None:raise RuntimeError(f'Unreadable input: {path}')
        mask=read_mask(args.mask,frame.shape)
        frame=transform_frame(frame,args.tx,args.ty,args.angle,args.scale) if (args.tx or args.ty or args.angle or args.scale!=1) else frame
        result=infer_map(frame,mask,enc,dec,device,args.patch_size,args.stride,args.min_coverage,args.batch_size,args.sigma)
        identifier=(f"{path['scenario']}:{path['ordinal']}" if args.dataset_source=='mcap' else str(path))
        rec={'file':identifier,'score':result['score'],'n_patches':len(result['boxes']),
             'covered_mask_fraction':result['coverage_ratio'],'tx':args.tx,'ty':args.ty,'angle':args.angle,'scale':args.scale}
        if tau is not None:rec.update(threshold=tau,anomalous=int(result['score']>tau))
        records.append(rec)
        if args.save_maps:save_map(out/f'{i:06d}_overlay.png',frame,mask,result)
        print(f'[{i+1}/{len(paths)}] {identifier} score={result["score"]:.6f} coverage={result["coverage_ratio"]:.3f}')
    with (out/'scores.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    if args.calibrate:
        scores=np.array([r['score'] for r in records]);tau=float(np.percentile(scores,args.threshold_percentile))
        config={'threshold':tau,'calibration_percentile':args.threshold_percentile,
                **{k:getattr(args,k) for k in ('patch_size','stride','min_coverage','sigma')},
                'scorer':'mean_absolute_residual_rgb_gaussian_stitched_q99',
                'n_calibration_frames':len(scores),'checkpoint':str(args.checkpoint)}
        (out/'threshold.json').write_text(json.dumps(config,indent=2))
        print('[calibrated]',tau,'saved',out/'threshold.json')
    elif tau is not None:
        print('[summary] anomalies',sum(r['anomalous'] for r in records),'/',len(records))
    else: print('[info] No threshold provided; scores only')
if __name__=='__main__':main()

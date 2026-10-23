#!/usr/bin/env python3
import argparse,sys,json,random,csv,time,hashlib,os,signal
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import cv2
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from r2ad.core import build_models
from r2ad.dataset import RectangleDataset
from r2ad.mcap_data import McapRectangleDataset
from r2ad.config import resolve, DEFAULT_CONFIG
from tqdm.auto import tqdm
from contextlib import nullcontext
from r2ad.training_state import atomic_save,capture_rng,restore_rng,ResumableBatches,PauseSignals


def initialize_worker(worker_id):
    signal.signal(signal.SIGINT,signal.SIG_IGN)
    signal.signal(signal.SIGTERM,signal.SIG_DFL)
    if hasattr(signal,'SIGUSR1'):signal.signal(signal.SIGUSR1,signal.SIG_IGN)
    cv2.setNumThreads(1)
    seed=torch.initial_seed() % (2**32)
    np.random.seed(seed);random.seed(seed)


def precision_context(device, precision):
    return torch.autocast(device_type='cuda',dtype=torch.bfloat16) if precision == 'bf16' else nullcontext()


def save_probe_grid(probe, enc, dec, device, path, random_probe=None, details=None, vmin=0., vmax=.5):
    """Each patch is a column; rows show input, reconstruction and error."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    was_enc, was_dec = enc.training, dec.training
    enc.eval(); dec.eval()
    with torch.inference_mode():
        inputs = torch.cat([probe, random_probe]) if random_probe is not None else probe
        x = inputs.to(device, non_blocking=True)
        mu, _ = enc(x)
        y = dec(mu)
        errs = (x - y).abs().mean(dim=1)
        x_np = ((x.detach().cpu().float().permute(0,2,3,1).numpy()+1)/2).clip(0,1)
        y_np = ((y.detach().cpu().float().permute(0,2,3,1).numpy()+1)/2).clip(0,1)
        e_np = (errs.detach().cpu().float().numpy()/2).clip(0,1)
    columns = len(x_np)
    fig, axes = plt.subplots(3, columns, figsize=(max(3*columns,4), 9), squeeze=False)
    for i in range(columns):
        axes[0,i].imshow(x_np[i]); axes[1,i].imshow(y_np[i]); axes[2,i].imshow(e_np[i],cmap='Reds',vmin=vmin,vmax=vmax)
        for ax in axes[:,i]:
            ax.set_xticks([]); ax.set_yticks([])
        title = f'Fixed validation {i+1}' if i < len(probe) else f'Fresh random train {i-len(probe)+1}'
        axes[0,i].set_title(title, fontsize=10)
    for j,t in enumerate(('Original patch','Reconstructed (mu)',f'Absolute error ({vmin:g}..{vmax:g})')):
        axes[j,0].set_ylabel(t,fontsize=10)
    if details:
        fig.suptitle(details, fontsize=11)
    fig.tight_layout(rect=(0,0,1,.94) if details else (0,0,1,1))
    path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,dpi=125)
    plt.close(fig)
    enc.train(was_enc); dec.train(was_dec)


def epoch_learning_metrics(loss_sums, counts, samples, kl_weight, adv_weight):
    reconstruction,kl,gan,discriminator,total=(loss_sums / samples).detach().cpu().tolist()
    tp,fn,tn,fp=counts.detach().cpu().tolist()
    return dict(reconstruction=reconstruction,kl=kl,generator_adversarial=gan,
                weighted_kl=kl_weight*kl,weighted_gan=adv_weight*gan,
                discriminator=discriminator,generator_total=total,
                discriminator_accuracy=(tp+tn)/max(tp+fn+tn+fp,1),
                discriminator_f1=2*tp/max(2*tp+fp+fn,1))


def save_training_dashboard(summary, path, partial=None):
    """Losses and real-vs-reconstruction diagnostics; never anomaly accuracy."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    records=summary['epochs'] + ([partial] if partial else [])
    if not records:
        return
    epochs=[row['epoch'] for row in records]
    fig,axes=plt.subplots(1,3,figsize=(17,5))
    for key,label,color in [('reconstruction','Train reconstruction','blue'),('validation_mse','Validation reconstruction','orange'),
                             ('kl','KL','teal'),('weighted_kl','Weighted KL','cyan'),
                             ('generator_adversarial','Generator GAN','green'),('weighted_gan','Weighted GAN','limegreen'),
                             ('generator_total','Generator total','red')]:
        values=np.array([row.get(key,float('nan')) for row in records])
        axes[0].plot(epochs,np.where(values>0,values,np.nan),label=label,color=color,marker='.' if len(records)==1 else None)
    axes[0].set(yscale='log',xlabel='Epoch',ylabel='Loss (log)',title='Reconstruction and generator objective')
    for key,label,marker in [('discriminator_accuracy','Real/fake accuracy','+'),('discriminator_f1','Real-class F1','x')]:
        axes[1].plot(epochs,[row[key] for row in records],marker=marker,label=label)
    axes[1].axhline(.5,color='gray',linestyle=':',label='50% accuracy reference')
    axes[1].set(xlabel='Epoch',ylabel='Discriminator score',ylim=(0,1.02),title='Real=1; reconstruction=0 (training)')
    for key,label,color in [('generator_adversarial','Generator GAN','green'),('discriminator','Discriminator BCE','purple')]:
        axes[2].plot(epochs,[row[key] for row in records],color=color,label=label,marker='.' if len(records)==1 else None)
    axes[2].set(xlabel='Epoch',ylabel='Loss',title='Adversarial balance')
    for ax in axes:
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    title=f"Learning progression | {summary['config']['safety_area']} | epoch {epochs[-1]}"
    if partial:
        title+=' (partial epoch; validation pending)'
    fig.suptitle(title)
    fig.tight_layout();path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,dpi=125);plt.close(fig)


def random_monitor_patches(dataset, count, seed):
    """Fresh training samples without consuming the training process's NumPy RNG."""
    state = np.random.get_state()
    try:
        np.random.seed(seed % (2**32))
        indices = np.random.randint(len(dataset), size=count)
        return torch.stack([dataset[int(index)] for index in indices])
    finally:
        np.random.set_state(state)


def duration(seconds):
    if seconds is None:
        return 'estimating'
    seconds = max(0, int(seconds))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'


def write_training_summary(summary, path):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    temporary.replace(path)
    data = summary['data']
    settings = summary['config']
    lines = ['# R2AD training report', '',
             f"Status: {summary['status']}. Started: {summary['started_at']}.", '',
             f"Configuration: `{settings['config']}`. Safety area: `{settings['safety_area']}`. Mask: `{settings['mask']}`.", '',
             f"Training input: `{settings.get('train_dir') or settings.get('manifest')}`. "
             f"Validation input: `{settings.get('val_dir') or settings.get('manifest')}`. Checkpoint: `{settings['output']}`.", '',
             f"Data: {data['train_frames']} normal training frames, {data['validation_frames']} normal validation frames. "
             f"Each epoch samples {data['train_patches_per_epoch']} training patches and {data['validation_patches_per_epoch']} validation patches.", '',
             f"Source patch size is approximately {data['source_patch_size']} pixels; network input is {data['network_input_size']}×{data['network_input_size']}. "
             'Training uses random rectangles with geometric/brightness augmentation; validation uses fresh random rectangles without augmentation.', '',
             'The generator learns with reconstruction, KL and adversarial losses; the discriminator distinguishes original and reconstructed training patches. '
             'Best-checkpoint selection uses validation reconstruction MSE, not test labels or anomaly scores.', '',
             f"Elapsed: {duration(summary['elapsed_seconds'])}. Estimated remaining: {duration(summary['estimated_remaining_seconds'])}. "
             f"Device: {summary['device']}. Precision: {summary.get('precision', 'fp32')}.", '', summary['timing_note'], '',
             f"Plan: {settings['epochs']} epochs, batch size {settings['batch_size']}, learning rate {settings['lr']}, "
             f"KL weight {settings['kl_weight']}, adversarial weight {settings['adv_weight']}, seed {settings['seed']}.", '',
             'Monitoring columns show fixed validation patches followed by fresh random augmented training patches. '
             'Rows show original, latent-mean reconstruction, and absolute error. Only fixed probes support comparisons of the same content over time.', '',
             '| Epoch | Train MSE | Validation MSE | Train time | Validation time | Epoch total | Patches/s | Loader wait | Best saved |',
             '|---|---|---|---|---|---|---|---|---|']
    for row in summary['epochs']:
        lines.append(f"| {row['epoch']} | {row['train_mse']:.6f} | {row['validation_mse']:.6f} | "
                     f"{duration(row['train_seconds'])} | {duration(row['validation_seconds'])} | "
                     f"{duration(row['epoch_seconds'])} | {row.get('patches_per_second', 0):.1f} | "
                     f"{duration(row.get('data_wait_seconds', 0))} | {row['checkpoint_saved']} |")
    if summary.get('best_epoch'):
        lines += ['', f"Best checkpoint: epoch {summary['best_epoch']}, validation MSE {summary['best_validation_mse']:.6f}."]
    path.with_name('training_report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def save_loss_plot(records, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,4))
    plotted=[records[i] for i in np.linspace(0,len(records)-1,min(2000,len(records)),dtype=int)]
    for key in ('reconstruction','kl','generator_adversarial','discriminator'):
        ax.plot([r['step'] for r in plotted],[r[key] for r in plotted],label=key,alpha=.8)
    ax.set(xlabel='Global optimizer steps',ylabel='Loss',title='R2AD training diagnostics')
    ax.legend(fontsize=8);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(path,dpi=125);plt.close(fig)


def save_metrics(records, path, written=0):
    with path.open('a' if written else 'w',newline='') as fp:
        w=csv.DictWriter(fp,fieldnames=('epoch','step','batch','reconstruction','kl','generator_adversarial','discriminator','generator_total'))
        if not written:
            w.writeheader()
        w.writerows(records[written:])
    return len(records)

def train(pause):
    run_started = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()
    p=argparse.ArgumentParser(description='Train R2AD VAE-GAN using normal full-resolution images')
    p.add_argument('--dataset-source',choices=('images','mcap'),default='images')
    p.add_argument('--manifest',help='MCAP manifest from index_rosbags_r2ad.py')
    p.add_argument('--config',default=str(DEFAULT_CONFIG));p.add_argument('--safety-area',choices=('PLeft','PRight','RoboArm','ConvBelt','SafetyZone'))
    for flag in ('train-dir','val-dir','mask','output'):p.add_argument('--'+flag)
    p.add_argument('--epochs',type=int);p.add_argument('--batch-size',type=int)
    p.add_argument('--patch-size',type=int,default=224);p.add_argument('--min-coverage',type=float,default=.85)
    p.add_argument('--samples-per-frame',type=int,default=6);p.add_argument('--latent-dims',type=int)
    p.add_argument('--lr',type=float,default=.0002);p.add_argument('--workers',type=int)
    p.add_argument('--kl-weight',type=float,default=.0001);p.add_argument('--adv-weight',type=float,default=.01)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--device',default='auto')
    p.add_argument('--precision',choices=('auto','fp32','bf16'),default='auto',help='auto uses BF16 on supported CUDA devices; losses remain FP32')
    p.add_argument('--prefetch-factor',type=int,default=2)
    p.add_argument('--persistent-workers',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--opencv-threads',type=int,default=1,help='OpenCV threads in the main process; workers use one each')
    p.add_argument('--augmentation-mode',choices=('crop','full'),default='crop',help='Render only selected image ROI, or use legacy full-frame rendering (image datasets)')
    p.add_argument('--tf32',action=argparse.BooleanOptionalAction,default=False)
    p.add_argument('--cudnn-benchmark',action=argparse.BooleanOptionalAction,default=False)
    p.add_argument('--monitor-every',type=int,default=1000,help='Save monitoring images every N optimizer steps; 0 disables')
    p.add_argument('--monitor-probes',type=int,default=4,help='Number of fixed validation patches to display')
    p.add_argument('--monitor-random-probes',type=int,default=2,help='Fresh augmented random training patches per monitoring image; 0 disables')
    p.add_argument('--monitor-dir',default=None,help='Defaults to checkpoint parent/live_monitor')
    p.add_argument('--save-cp-k','--save_cp_k',type=int,default=0,
                   help='Save a separate checkpoint every K completed epochs; 0 disables (best model is always saved)')
    p.add_argument('--resume',type=Path,help='Continue a resumable checkpoint; legacy model snapshots load weights with fresh optimizers')
    p.add_argument('--save-every-steps',type=int,default=1000,help='Refresh last_checkpoint.pt every N optimizer steps; 0 means epoch/pause saves only')
    p.add_argument('--map-vmin',type=float,default=0.)
    p.add_argument('--map-vmax',type=float,default=.5,help='Fixed Reds error-map maximum on RGB 0..1 residual scale')
    args=p.parse_args()
    resume=None
    if args.resume:
        resume=torch.load(args.resume,map_location='cpu',weights_only=False)
        saved=resume.get('config',{})
        supplied={token.split('=')[0].replace('-','_').lstrip('_') for token in sys.argv[1:] if token.startswith('--')}
        for name,value in saved.items():
            if hasattr(args,name) and name not in supplied and name not in ('resume','save_every_steps'):
                setattr(args,name,value)
        args.latent_dims=resume.get('latent_dims',args.latent_dims)
        if resume.get('training_state'):
            for name in ('dataset_source','safety_area','latent_dims','batch_size','patch_size','min_coverage',
                         'samples_per_frame','augmentation_mode','kl_weight','adv_weight','lr','seed','train_dir','val_dir','manifest','mask'):
                if name in saved and getattr(args,name,None)!=saved[name]:
                    p.error(f'Full resume requires unchanged {name}; checkpoint={saved[name]!r}')
    if not args.safety_area:
        p.error('--safety-area is required for a new run (or must be present in resume metadata)')
    if args.dataset_source=='mcap':
        from r2ad.config import load_config
        cfg=load_config(args.config);root=Path(cfg['data']['dataset_base']).expanduser();rcfg=cfg.get('r2ad') or {}
        mask_cfg=(rcfg.get('safety_zone_mask') or cfg.get('data',{}).get('mask_types',{}).get('SafetyZone')) if args.safety_area=='SafetyZone' else cfg['data']['mask_types'].get(args.safety_area)
        args.mask=args.mask or mask_cfg
        if not args.mask:raise ValueError('Missing safety mask in config')
        args.mask=str((root/args.mask) if not Path(args.mask).is_absolute() else args.mask)
        args.manifest=args.manifest or rcfg.get('manifest') or str(root/'r2ad'/'manifest.csv')
        args.manifest=str((root/args.manifest) if not Path(args.manifest).is_absolute() else args.manifest)
        base=Path(cfg.get('results',{}).get('output_dir','results'))/str(rcfg.get('dataset_version','V6'))/'r2ad'/args.safety_area
        args.output=args.output or str(base/'model_r2ad.pt')
        args.latent_dims=args.latent_dims or int(cfg.get('models',{}).get('latent_dims',64))
        args.epochs=args.epochs or int(cfg.get('training',{}).get('epochs',50))
        args.batch_size=args.batch_size or int(cfg.get('training',{}).get('batch_size',32))
        args.workers=args.workers if args.workers is not None else int(cfg.get('training',{}).get('num_workers',4))
        if not Path(args.manifest).is_file():raise FileNotFoundError(args.manifest)
        if not Path(args.mask).is_file():raise FileNotFoundError(args.mask)
    else:args=resolve(args,'train')
    if args.epochs < 1 or args.batch_size < 1 or args.samples_per_frame < 1:
        p.error('epochs, batch-size and samples-per-frame must be positive')
    if args.save_cp_k < 0 or args.save_every_steps < 0:
        p.error('--save-cp-k and --save-every-steps must be >= 0')
    if args.monitor_every < 0 or args.monitor_probes < 1 or args.monitor_random_probes < 0:
        p.error('Require monitor-every >= 0, monitor-probes >= 1 and monitor-random-probes >= 0')
    if args.workers < 0 or args.prefetch_factor < 1 or args.opencv_threads < 0:
        p.error('Require workers >= 0, prefetch-factor >= 1, opencv-threads >= 0')
    if not 0 <= args.map_vmin < args.map_vmax <= 1:
        p.error('Require 0 <= map-vmin < map-vmax <= 1')
    cv2.setNumThreads(args.opencv_threads)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    device=('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
    is_cuda=torch.device(device).type == 'cuda'
    if is_cuda and torch.device(device).index is not None:
        torch.cuda.set_device(torch.device(device).index)
    bf16_supported=is_cuda and torch.cuda.is_bf16_supported()
    if args.precision == 'bf16' and not bf16_supported:
        p.error('BF16 requires a supported CUDA device; use --precision fp32')
    precision='bf16' if (args.precision == 'bf16' or args.precision == 'auto' and bf16_supported) else 'fp32'
    if is_cuda:
        torch.backends.cuda.matmul.allow_tf32=args.tf32
        torch.backends.cudnn.allow_tf32=args.tf32
        torch.backends.cudnn.benchmark=args.cudnn_benchmark
    if args.dataset_source=='mcap':
        train=McapRectangleDataset(args.manifest,'train',args.mask,args.patch_size,args.min_coverage,args.samples_per_frame,True,args.seed)
        val=McapRectangleDataset(args.manifest,'val',args.mask,args.patch_size,args.min_coverage,1,False,args.seed)
    else:
        train=RectangleDataset(args.train_dir,args.mask,args.patch_size,args.min_coverage,args.samples_per_frame,True,args.seed,args.augmentation_mode)
        val=RectangleDataset(args.val_dir,args.mask,args.patch_size,args.min_coverage,1,False,args.seed)
    digest=hashlib.sha256()
    for dataset in (train,val):
        digest.update(json.dumps(dataset.rows if hasattr(dataset,'rows') else [str(path.resolve()) for path in dataset.paths],sort_keys=True).encode())
    for value in (args.mask,getattr(args,'train_dir',None),getattr(args,'val_dir',None),args.manifest):
        if value and Path(value).is_file():digest.update(Path(value).read_bytes())
    data_signature=digest.hexdigest()
    if resume and resume.get('training_state',{}).get('data_signature',data_signature)!=data_signature:
        p.error('Training/validation manifest, image inventory or mask changed since checkpoint')
    loader_options=dict(num_workers=args.workers,pin_memory=is_cuda,worker_init_fn=initialize_worker,
                        generator=torch.Generator().manual_seed(args.seed+2468))
    if args.workers:
        loader_options.update(persistent_workers=args.persistent_workers,prefetch_factor=args.prefetch_factor)
    sampler=ResumableBatches(len(train),args.batch_size,args.seed)
    total_batches=(len(train)+args.batch_size-1)//args.batch_size
    tr=DataLoader(train,batch_sampler=sampler,**loader_options)
    va=DataLoader(val,batch_size=args.batch_size,shuffle=False,**loader_options)
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    train_frames = len(train.rows) if args.dataset_source == 'mcap' else len(train.paths)
    val_frames = len(val.rows) if args.dataset_source == 'mcap' else len(val.paths)
    summary = dict(status='initializing', started_at=started_at, config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}, device=device,precision=precision,
                   data=dict(train_frames=train_frames, validation_frames=val_frames,
                             train_patches_per_epoch=len(train), validation_patches_per_epoch=len(val),
                             train_batches_per_epoch=total_batches, validation_batches_per_epoch=len(va),
                             source_patch_size=args.patch_size, network_input_size=128,
                             min_coverage=args.min_coverage, samples_per_frame=args.samples_per_frame),
                   epochs=[], elapsed_seconds=0, estimated_remaining_seconds=None,
                   timing_note='ETA includes measured epoch overhead after the first epoch; first-epoch ETA estimates training batches only.')
    summary_path=out.parent/'training_summary.json'
    start_epoch=1;resume_batch=0;resume_epoch_state=None;resume_order=None
    if resume and resume.get('training_state'):
        state=resume['training_state']
        start_epoch=state['next_epoch'];resume_batch=state['next_batch']
        resume_epoch_state=state.get('epoch_state');resume_order=state.get('order')
        previous=state['summary']
        run_started-=previous.get('elapsed_seconds',0)
        summary.update(epochs=previous['epochs'],started_at=previous['started_at'])
        for key in ('best_epoch','best_validation_mse'):
            if key in previous:summary[key]=previous[key]
    elif resume:
        start_epoch=int(resume.get('epoch',0))+1
        print('[resume] Legacy weights-only snapshot: optimizer and RNG state are unavailable; this is a warm start.',flush=True)
    if args.epochs < start_epoch:
        p.error(f'--epochs is the TOTAL target; set it >= {start_epoch} to continue')
    write_training_summary(summary, summary_path)
    print(f'[data] source={args.dataset_source} area={args.safety_area} mask={args.mask}',flush=True)
    print(f'[data] train frames={train_frames}, val frames={val_frames}; '
          f'patches/epoch={len(train)} train + {len(val)} validation; '
          f'batches/epoch={total_batches} train + {len(va)} validation',flush=True)
    print(f'[plan] epochs={args.epochs} total optimizer steps={args.epochs*total_batches} '
          f'device={device} source patch~{args.patch_size}px -> network=128px; '
          'normal-only training, fresh random patches and augmentation; validation uses random unaugmented patches',flush=True)
    print(f'[output] checkpoint={out} summary={summary_path}',flush=True)
    print(f'[control] pid={os.getpid()} | Ctrl+C or SIGUSR1/SIGTERM requests pause; '
          f'latest={out.parent/"last_checkpoint.pt"} | recovery interval={args.save_every_steps} steps + every epoch',flush=True)
    print(f'[performance] precision={precision} workers={args.workers} persistent={bool(args.workers and args.persistent_workers)} '
          f'prefetch={args.prefetch_factor if args.workers else 0} augmentation={args.augmentation_mode if args.dataset_source == "images" else "full (MCAP)"} '
          f'TF32={args.tf32} cudnn_benchmark={args.cudnn_benchmark}',flush=True)
    enc,dec,disc=build_models(args.latent_dims,device)
    opt_g=torch.optim.Adam(list(enc.parameters())+list(dec.parameters()),lr=args.lr)
    opt_d=torch.optim.Adam(disc.parameters(),lr=args.lr/2)
    bce=nn.BCEWithLogitsLoss();best=float('inf')
    mon_dir=Path(args.monitor_dir) if args.monitor_dir else out.parent/'live_monitor'
    history=[]; global_step=0; probe=None;written_metrics=0
    if resume:
        for model,key in [(enc,'encoder_state_dict'),(dec,'decoder_state_dict'),(disc,'discriminator_state_dict')]:
            model.load_state_dict(resume[key])
        best=float(resume.get('best_val_mse',resume.get('val_mse',float('inf'))))
        if resume.get('training_state'):
            opt_g.load_state_dict(resume['optimizer_g_state_dict']);opt_d.load_state_dict(resume['optimizer_d_state_dict'])
            global_step=resume['training_state']['global_step']
            history=resume['training_state'].get('history',[])
            probe=resume['training_state'].get('fixed_probes')
        else:
            global_step=(start_epoch-1)*total_batches
            summary.update(best_epoch=resume.get('epoch'),best_validation_mse=best)
        print(f'[resume] {args.resume}: epoch={start_epoch} completed_batches={resume_batch} optimizer_step={global_step}',flush=True)
    latest=out.parent/'last_checkpoint.pt'
    def save_resume(epoch,batch,epoch_state=None,complete=False,status='training'):
        summary.update(status=status,elapsed_seconds=time.monotonic()-run_started,optimizer_steps=global_step)
        snapshot=dict(encoder_state_dict=enc.state_dict(),decoder_state_dict=dec.state_dict(),
                      discriminator_state_dict=disc.state_dict(),optimizer_g_state_dict=opt_g.state_dict(),
                      optimizer_d_state_dict=opt_d.state_dict(),latent_dims=args.latent_dims,
                      epoch=epoch,val_mse=summary['epochs'][-1]['validation_mse'] if complete else None,best_val_mse=best,
                      config=summary['config'],model_kind='r2ad_rectangles',checkpoint_version=2,
                      training_state=dict(next_epoch=epoch+1 if complete else epoch,next_batch=0 if complete else batch,
                          epoch_state=epoch_state,order=None if complete else sampler.order,global_step=global_step,
                          rng=capture_rng(),summary=summary,history=history,fixed_probes=probe,data_signature=data_signature))
        atomic_save(snapshot,latest)
        write_training_summary(summary,summary_path)
        return snapshot
    if args.monitor_every:
        mon_dir.mkdir(parents=True,exist_ok=True)
        # The same patches are held in memory for the entire run.
        probe_count=min(args.monitor_probes,len(val))
        if probe is None:probe=random_monitor_patches(val,probe_count,args.seed)
        if history:written_metrics=save_metrics(history,mon_dir/'batch_losses.csv')
        torch.save(probe,mon_dir/'fixed_probes.pt')
        fresh=random_monitor_patches(train,args.monitor_random_probes,args.seed+1) if args.monitor_random_probes else None
        save_probe_grid(probe,enc,dec,device,mon_dir/f'probe_step_{global_step:06d}.png',fresh,
                        'Before training | fixed validation probes + fresh augmented training patches',args.map_vmin,args.map_vmax)
        print(f'[monitor] fixed probes={probe_count} random train probes={args.monitor_random_probes} | output={mon_dir}',flush=True)
    setup_seconds=time.monotonic()-run_started
    summary.update(status='training', setup_seconds=setup_seconds)
    if resume and resume.get('training_state'):restore_rng(resume['training_state']['rng'])
    for epoch in tqdm(range(start_epoch,args.epochs+1),desc='Epochs'):
        continuing=epoch==start_epoch and resume_epoch_state is not None
        saved_epoch=resume_epoch_state if continuing else {}
        batch_offset=resume_batch if epoch==start_epoch else 0
        sampler.set_epoch(epoch,resume_order if epoch==start_epoch else None,batch_offset)
        epoch_started=time.monotonic()
        epoch_started-=saved_epoch.get('train_seconds',0)
        warmup_finished=None;warmup_batches=batch_offset+min(10,max(0,total_batches-batch_offset-1))
        data_wait_seconds=saved_epoch.get('data_wait_seconds',0.);last_batch_end=time.monotonic();samples_processed=saved_epoch.get('samples_processed',0)
        enc.train();dec.train();disc.train();running=[]
        loss_sums=torch.zeros(5,device=device);discriminator_counts=torch.zeros(4,device=device)
        if continuing:
            loss_sums=saved_epoch['loss_sums'].to(device);discriminator_counts=saved_epoch['counts'].to(device)
        def partial_state():
            return dict(train_seconds=time.monotonic()-epoch_started,data_wait_seconds=data_wait_seconds,
                        samples_processed=samples_processed,loss_sums=loss_sums.detach().cpu(),counts=discriminator_counts.detach().cpu())
        completed_batch=batch_offset
        batches=tqdm(pause_aware_batches(tr,pause),total=len(tr),desc=f'Epoch {epoch} batches',leave=False)
        for batch_idx,x in enumerate(batches,start=batch_offset+1):
            data_wait_seconds+=time.monotonic()-last_batch_end
            samples_processed+=len(x)
            x=x.to(device,non_blocking=True)
            with precision_context(device,precision):
                mu,logvar=enc(x)
                # Keep variance exponentiation and KL arithmetic in FP32.
                z=mu.float()+torch.randn_like(mu.float())*torch.exp(.5*logvar.float())
                recon=dec(z)
            opt_d.zero_grad(set_to_none=True)
            with precision_context(device,precision):
                logits_real=disc(x)
                logits_fake=disc(recon.detach())
            logits_real=logits_real.float();logits_fake=logits_fake.float()
            dloss=.5*(bce(logits_real,torch.ones_like(logits_real))+bce(logits_fake,torch.zeros_like(logits_fake)))
            dloss.backward();opt_d.step()
            for par in disc.parameters():par.requires_grad_(False)
            opt_g.zero_grad(set_to_none=True)
            rloss=F.mse_loss(recon.float(),x)
            kl=-.5*(1+logvar.float()-mu.float().square()-logvar.float().exp()).sum(dim=1).mean()/args.latent_dims
            with precision_context(device,precision):
                pred_fake=disc(recon)
            pred_fake=pred_fake.float()
            adv=bce(pred_fake,torch.ones_like(pred_fake))
            gloss=rloss+args.kl_weight*kl+args.adv_weight*adv
            gloss.backward();opt_g.step()
            for par in disc.parameters():par.requires_grad_(True)
            running.append(float(rloss.item()))
            loss_sums+=torch.stack([rloss.detach(),kl.detach(),adv.detach(),dloss.detach(),gloss.detach()])*len(x)
            discriminator_counts+=torch.stack([(logits_real.detach()>=0).sum(),(logits_real.detach()<0).sum(),
                                               (logits_fake.detach()<0).sum(),(logits_fake.detach()>=0).sum()])
            global_step+=1
            completed_batch=batch_idx
            if args.monitor_every:
                rec=dict(epoch=epoch,step=global_step,batch=batch_idx,
                         reconstruction=float(rloss.detach()),kl=float(kl.detach()),
                         generator_adversarial=float(adv.detach()),
                         discriminator=float(dloss.detach()),generator_total=float(gloss.detach()))
                history.append(rec)
                if global_step==1 or global_step % args.monitor_every==0:
                    fresh=random_monitor_patches(train,args.monitor_random_probes,args.seed+global_step+1) if args.monitor_random_probes else None
                    save_probe_grid(probe,enc,dec,device,mon_dir/f'probe_step_{global_step:06d}.png',fresh,
                                    f'Epoch {epoch}/{args.epochs} | step {global_step} | elapsed {duration(time.monotonic()-run_started)}',args.map_vmin,args.map_vmax)
                    written_metrics=save_metrics(history,mon_dir/'batch_losses.csv',written_metrics)
                    save_loss_plot(history,mon_dir/'batch_losses.png')
                    partial=dict(epoch=epoch,**epoch_learning_metrics(loss_sums,discriminator_counts,samples_processed,args.kl_weight,args.adv_weight))
                    save_training_dashboard(summary,out.parent/'training_progress.png',partial)
                    print(f'[monitor step={global_step}] rec={rec["reconstruction"]:.5f} '
                          f'KL={rec["kl"]:.5f} D={rec["discriminator"]:.5f} '
                          f'G={rec["generator_total"]:.5f} | {mon_dir}',flush=True)
            if batch_idx == warmup_batches:
                warmup_finished=time.monotonic()
            if batch_idx == 1 or batch_idx % 100 == 0 or batch_idx == total_batches:
                batch_seconds=((time.monotonic()-warmup_finished)/(batch_idx-warmup_batches)
                               if warmup_finished is not None and batch_idx > warmup_batches else None)
                epoch_overhead=(np.mean([r['epoch_seconds']-r['train_seconds'] for r in summary['epochs']])
                                if summary['epochs'] else 0)
                remaining_steps=(total_batches-batch_idx)+(args.epochs-epoch)*total_batches
                eta=float(batch_seconds*remaining_steps+epoch_overhead*(args.epochs-epoch+1)) if batch_seconds is not None else None
                elapsed=time.monotonic()-run_started
                batches.set_postfix(rec=f'{running[-1]:.5f}',elapsed=duration(elapsed),run_ETA=duration(eta))
                print(f'[progress] epoch={epoch}/{args.epochs} batch={batch_idx}/{total_batches} '
                      f'step={global_step}/{args.epochs*total_batches} elapsed={duration(elapsed)} '
                      f'patches/s={samples_processed/max(time.monotonic()-epoch_started,1e-9):.1f} '
                      f'run_ETA~{duration(eta)}'+(' (warming up)' if eta is None else ' (training-only estimate until first epoch completes)' if not summary['epochs'] else ''),flush=True)
                summary.update(elapsed_seconds=elapsed, estimated_remaining_seconds=eta,
                               current_epoch=epoch, optimizer_steps=global_step)
                write_training_summary(summary,summary_path)
            if pause.requested or args.save_every_steps and global_step % args.save_every_steps==0:
                save_resume(epoch,batch_idx,partial_state(),status='paused' if pause.requested else 'training')
                if pause.requested:
                    print(f'[paused] Saved {latest}; resume with --resume {latest}',flush=True)
                    return
            last_batch_end=time.monotonic()
        if pause.requested:
            save_resume(epoch,completed_batch,partial_state(),status='paused')
            print(f'[paused] Saved {latest}',flush=True)
            return
        train_seconds=time.monotonic()-epoch_started
        validation_started=time.monotonic()
        print(f'[validation] epoch={epoch} batches={len(va)}',flush=True)
        enc.eval();dec.eval();vals=[]
        with torch.inference_mode():
            for x in tqdm(pause_aware_batches(va,pause),total=len(va),desc=f'Epoch {epoch} validation',leave=False):
                if pause.requested:
                    state=partial_state();state['train_seconds']=train_seconds
                    save_resume(epoch,total_batches,state,status='paused')
                    print(f'[paused] Saved {latest}; validation restarts on resume.',flush=True)
                    return
                x=x.to(device,non_blocking=True)
                with precision_context(device,precision):
                    mu,_=enc(x);recon=dec(mu)
                vals.append((F.mse_loss(recon.float(),x,reduction='none').mean((1,2,3))).cpu())
        if pause.requested:
            state=partial_state();state['train_seconds']=train_seconds
            save_resume(epoch,total_batches,state,status='paused')
            print(f'[paused] Saved {latest}; validation restarts on resume.',flush=True)
            return
        v=float(torch.cat(vals).mean());t=float(loss_sums[0].item()/samples_processed)
        validation_seconds=time.monotonic()-validation_started
        print(f'[epoch {epoch:03d}] train_rec={t:.6f} val_mse={v:.6f}')
        if args.monitor_every:
            fresh=random_monitor_patches(train,args.monitor_random_probes,args.seed+global_step+epoch+1) if args.monitor_random_probes else None
            save_probe_grid(probe,enc,dec,device,mon_dir/f'probe_epoch_{epoch:03d}.png',fresh,
                            f'Epoch {epoch}/{args.epochs} complete | train MSE={t:.6f} val MSE={v:.6f}',args.map_vmin,args.map_vmax)
            written_metrics=save_metrics(history,mon_dir/'batch_losses.csv',written_metrics)
            save_loss_plot(history,mon_dir/'batch_losses.png')
        improved=v<best
        periodic_path=None
        if args.save_cp_k and epoch % args.save_cp_k == 0:
            periodic_path=out.parent/'checkpoints'/f'{out.stem}_epoch_{epoch:04d}{out.suffix or ".pt"}'
        if improved:
            best=v
            summary['best_epoch']=epoch
        epoch_seconds=time.monotonic()-epoch_started
        summary['epochs'].append(dict(epoch=epoch,train_mse=t,validation_mse=v,
                                      **epoch_learning_metrics(loss_sums,discriminator_counts,samples_processed,args.kl_weight,args.adv_weight),
                                      train_seconds=train_seconds,validation_seconds=validation_seconds,
                                      patches_per_second=samples_processed/max(train_seconds,1e-9),
                                      data_wait_seconds=data_wait_seconds,
                                      epoch_seconds=epoch_seconds,checkpoint_saved=improved,
                                      periodic_checkpoint=str(periodic_path) if periodic_path else ''))
        save_training_dashboard(summary,out.parent/'training_progress.png')
        epoch_seconds=time.monotonic()-epoch_started
        summary['epochs'][-1]['epoch_seconds']=epoch_seconds
        elapsed=time.monotonic()-run_started
        eta=float(np.mean([r['epoch_seconds'] for r in summary['epochs']])*(args.epochs-epoch))
        summary.update(elapsed_seconds=elapsed, estimated_remaining_seconds=eta,best_validation_mse=best)
        write_training_summary(summary,summary_path)
        with (out.parent/'epoch_metrics.csv').open('w',newline='') as fp:
            writer=csv.DictWriter(fp,fieldnames=list(summary['epochs'][0]))
            writer.writeheader();writer.writerows(summary['epochs'])
        snapshot=save_resume(epoch,total_batches,complete=True,
                             status='paused' if pause.requested else 'completed' if epoch==args.epochs else 'training')
        snapshot['val_mse']=v
        if improved:
            atomic_save(snapshot,out)
            print('[saved best]',out,flush=True)
        if periodic_path is not None:
            atomic_save(snapshot,periodic_path)
            print('[saved periodic]',periodic_path,flush=True)
        if pause.requested:
            print(f'[paused] Saved {latest}',flush=True)
            return
        print(f'[timing] epoch={duration(epoch_seconds)} train={duration(train_seconds)} '
              f'patches/s={samples_processed/max(train_seconds,1e-9):.1f} loader_wait={duration(data_wait_seconds)} '
              f'validation={duration(validation_seconds)} total={duration(elapsed)} '
              f'remaining~{duration(eta)} projected_total~{duration(elapsed+eta)}',flush=True)
    summary.update(status='completed',completed_at=datetime.now(timezone.utc).isoformat(),
                   elapsed_seconds=time.monotonic()-run_started,estimated_remaining_seconds=0)
    write_training_summary(summary,summary_path)
    print(f'[done] best validation MSE={best:.6f} best_epoch={summary["best_epoch"]} '
          f'total_time={duration(summary["elapsed_seconds"])} summary={summary_path}',flush=True)
def main():
    with PauseSignals() as pause:
        return train(pause)


def pause_aware_batches(loader,pause):
    iterator=iter(loader)
    while not pause.requested:
        try:
            batch=next(iterator)
        except StopIteration:
            return
        except RuntimeError:
            if pause.requested:return
            raise
        yield batch


if __name__=='__main__':main()

"""Resolve R2AD settings from the EXISTING ADVIS YAML. No production config overwritten."""
from pathlib import Path
import yaml

DEFAULT_CONFIG = Path('configs/cf_dataset_epito.yaml')
AREAS = ('PLeft', 'PRight', 'RoboArm', 'ConvBelt', 'SafetyZone')


def load_config(path):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Configuration not found: {path}')
    cfg = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
    if not isinstance(cfg, dict):
        raise ValueError('Expected YAML mapping at root')
    return cfg


def resolve(args, purpose):
    """Supply CLI defaults using data/models/results/r2ad YAML keys.

    Original ADVIS data.training/testing point to 128x128 masked crops, NOT full frames.
    Do not silently use those directories for R2AD.
    """
    cfg = load_config(args.config)
    area = args.safety_area
    if area not in AREAS:
        raise ValueError(f'Unknown safety area {area!r}; choose from {AREAS}')
    data = cfg.get('data', {})
    r2ad = cfg.get('r2ad', {})
    model_cfg = cfg.get('models', {})
    root = Path(data.get('dataset_base', '.')).expanduser()
    mask_from_cfg = (r2ad.get('safety_zone_mask') or data.get('mask_types', {}).get('SafetyZone')) if area == 'SafetyZone' else data.get('mask_types', {}).get(area)
    if mask_from_cfg and not Path(mask_from_cfg).is_absolute():
        mask_from_cfg = str(root / mask_from_cfg)
    args.mask = args.mask or mask_from_cfg
    if not args.mask:
        raise ValueError(f'No mask for {area} in data.mask_types, and no --mask override')
    args.mask = str(Path(args.mask).expanduser())
    frame_key = {'train': 'train_frames', 'calibrate': 'val_frames',
                 'evaluate': 'test_frames', 'stress': 'normal_test_frames'}[purpose]
    frame_argument = 'train_dir' if purpose == 'train' else 'input'
    frame_default = r2ad.get(frame_key)
    if frame_default and not Path(frame_default).is_absolute():
        frame_default = str(root / frame_default)
    frame_value = getattr(args, frame_argument, None) or frame_default
    if not frame_value:
        raise ValueError(
            f'Missing full-frame path: set r2ad.{frame_key} in {args.config}, '
            f'or provide --{frame_argument.replace("_", "-")}. '
            'Existing data.training/testing contain 128x128 area crops and cannot be used for R2AD.')
    setattr(args, frame_argument, str(Path(frame_value).expanduser()))
    if purpose == 'train':
        val_dir = args.val_dir or r2ad.get('val_frames')
        if val_dir and not Path(val_dir).is_absolute(): val_dir = str(root / val_dir)
        if not val_dir:
            raise ValueError('Set r2ad.val_frames in YAML or supply --val-dir (full-resolution normal frames)')
        args.val_dir = str(Path(val_dir).expanduser())
        if Path(args.train_dir).resolve() == Path(args.val_dir).resolve():
            raise ValueError('Train and validation directories must be different')
    result_base = Path(cfg.get('results', {}).get('output_dir', 'results'))
    result_folder = result_base / str(r2ad.get('dataset_version', root.name)) / 'r2ad' / area
    model_file = result_folder / 'model_r2ad.pt'
    output_dir = {
        'calibrate': result_folder / 'calibration',
        'evaluate': result_folder / 'test',
        'stress': result_folder / 'robustness',
    }
    if purpose == 'train':
        args.output = str(Path(args.output or model_file))
        if getattr(args, 'latent_dims', None) is None:
            args.latent_dims = int(model_cfg.get('latent_dims', 64))
        if getattr(args, 'epochs', None) is None:
            args.epochs = int(cfg.get('training', {}).get('epochs', 50))
        if getattr(args, 'batch_size', None) is None:
            args.batch_size = int(cfg.get('training', {}).get('batch_size', 32))
        if getattr(args, 'workers', None) is None:
            args.workers = int(cfg.get('training', {}).get('num_workers', 4))
    else:
        args.checkpoint = str(Path(args.checkpoint or model_file))
        args.output_dir = str(Path(args.output_dir or output_dir[purpose]))
        if hasattr(args, 'threshold_json') and not args.threshold_json and purpose != 'calibrate':
            args.threshold_json = str(result_folder / 'calibration' / 'threshold.json')
    if not Path(args.mask).is_file():
        raise FileNotFoundError(f'Mask missing: {args.mask}')
    for name in ([frame_argument] + (['val_dir'] if purpose == 'train' else [])):
        if not Path(getattr(args, name)).exists():
            raise FileNotFoundError(f'{name} does not exist: {getattr(args, name)}')
    return args

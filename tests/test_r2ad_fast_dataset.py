import cv2
import numpy as np
import torch
import pytest

from r2ad.dataset import RectangleDataset


def test_crop_rendering_matches_full_and_caches_mask(tmp_path, monkeypatch):
    # Nonuniform pixels and an irregular mask exercise actual geometry and borders.
    image = np.random.default_rng(1).integers(0, 256, (360, 480, 3), dtype=np.uint8)
    root = tmp_path / 'frames'
    root.mkdir()
    cv2.imwrite(str(root / 'frame.png'), image)
    mask = np.zeros((360, 480), np.uint8)
    mask[20:330, 30:450] = 255
    mask[20:90, 30:90] = 0
    mask_path = tmp_path / 'mask.png'
    cv2.imwrite(str(mask_path), mask)
    from r2ad import dataset as module
    original = module.read_mask
    calls = []
    def counted(*args):
        calls.append(1)
        return original(*args)
    monkeypatch.setattr(module, 'read_mask', counted)
    fast = RectangleDataset(root, mask_path, patch=128, augmentation_mode='crop')
    legacy = RectangleDataset(root, mask_path, patch=128, augmentation_mode='full')
    for seed in range(4):
        np.random.seed(seed)
        expected = legacy[0]
        np.random.seed(seed)
        actual = fast[0]
        # OpenCV coordinate arithmetic may differ slightly at interpolation boundaries.
        assert (actual - expected).abs().mean() < .002
        assert actual.shape == expected.shape == (3, 128, 128)
    assert len(calls) == 2  # Once per dataset/worker, rather than once per sample.
    assert np.array_equal(fast._mask_cache[(360, 480)], (mask > 127).astype(np.uint8))


def test_no_augmentation_is_identical(tmp_path):
    cv2.imwrite(str(tmp_path / 'frame.png'), np.full((128, 128, 3), 100, np.uint8))
    # Keep mask out of the image source directory.
    root = tmp_path / 'images'
    root.mkdir()
    (tmp_path / 'frame.png').rename(root / 'frame.png')
    mask = tmp_path / 'mask.png'
    cv2.imwrite(str(mask), np.full((128, 128), 255, np.uint8))
    a = RectangleDataset(root, mask, patch=64, augment=False, augmentation_mode='crop')
    b = RectangleDataset(root, mask, patch=64, augment=False, augmentation_mode='full')
    np.random.seed(4)
    first = a[0]
    np.random.seed(4)
    assert torch.equal(first, b[0])


def test_augmentation_controls_zero_and_gentle_ranges():
    from r2ad.dataset import sample_training_patch,validate_augmentation
    frame=np.random.default_rng(7).integers(0,256,(240,320,3),dtype=np.uint8)
    mask=np.ones((240,320),np.uint8)
    zero=dict(aug_translate=0,aug_rotate=0,aug_contrast=0,aug_brightness=0,patch_scale_min=1,patch_scale_max=1)
    _,info=sample_training_patch(frame,mask,64,.85,np.random.default_rng(10),details=True,settings=zero)
    assert np.array_equal(info['scene'],frame)
    assert info['augmentation']==dict(tx=0.,ty=0.,angle=0.,alpha=1.,beta=0.)
    assert info['box'][2:]==(64,64)
    gentle=dict(aug_translate=.005,aug_rotate=.5,aug_contrast=.05,aug_brightness=4,patch_scale_min=.98,patch_scale_max=1.02)
    for seed in range(8):
        _,info=sample_training_patch(frame,mask,64,.85,np.random.default_rng(seed),details=True,settings=gentle)
        a=info['augmentation']
        assert abs(a['tx'])<=1.6 and abs(a['ty'])<=1.2 and abs(a['angle'])<=.5
        assert .95<=a['alpha']<=1.05 and abs(a['beta'])<=4
        assert round(64*.98)<=info['box'][2]<=round(64*1.02)
    with pytest.raises(ValueError):validate_augmentation(dict(aug_translate=-.1))
    with pytest.raises(ValueError):validate_augmentation(dict(patch_scale_min=1.2,patch_scale_max=1.))

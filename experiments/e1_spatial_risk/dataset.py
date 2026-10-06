from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset


class SpatialRiskWindowDataset(Dataset):
    """Windowed E1 dataset: past risk maps -> future risk maps.

    Each item is one safety area in one scenario, so 128x128 coordinates always
    remain within a consistent physical safety-area crop.
    """

    def __init__(
        self,
        files: Iterable[str | Path],
        history: int = 15,
        horizons: tuple[int, ...] = (3, 5, 10, 15),
        stride: int = 1,
    ):
        self.files = [Path(p) for p in files]
        self.history = int(history)
        self.horizons = tuple(int(h) for h in horizons)
        self.stride = int(stride)
        if self.history < 1 or not self.horizons or min(self.horizons) < 1:
            raise ValueError("history and horizons must be positive")
        self.samples = []
        self.archives = []
        max_h = max(self.horizons)
        for file_idx, path in enumerate(self.files):
            data = np.load(path, allow_pickle=False)
            maps = data["risk_maps"]
            if maps.ndim != 4:
                raise ValueError(f"{path}: expected [T,A,H,W], got {maps.shape}")
            areas = [str(x) for x in data["areas"].tolist()]
            metadata = json.loads(str(data["metadata_json"].item()))
            self.archives.append({
                "path": path,
                "maps": maps,
                "areas": areas,
                "metadata": metadata,
            })
            T, A = maps.shape[:2]
            last_start = T - self.history - max_h
            for area_idx in range(A):
                for start in range(0, last_start + 1, self.stride):
                    self.samples.append((file_idx, area_idx, start))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        file_idx, area_idx, start = self.samples[index]
        archive = self.archives[file_idx]
        maps = archive["maps"]
        stop = start + self.history
        x = maps[start:stop, area_idx].astype(np.float32)[:, None, :, :]
        y = np.stack(
            [maps[stop + h - 1, area_idx] for h in self.horizons], axis=0
        ).astype(np.float32)[:, None, :, :]
        return {
            "x": torch.from_numpy(x),          # [T,1,H,W]
            "y": torch.from_numpy(y),          # [K,1,H,W]
            "scenario": archive["metadata"].get("scenario", archive["path"].stem),
            "area": archive["areas"][area_idx],
            "start": start,
        }

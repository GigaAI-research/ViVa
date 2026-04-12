# Viva Dataset
# 7-frame structure: condition frames + future_state + value (no future camera images)
# [0] blank, [1] state, [2] cam_left, [3] cam_right, [4] cam_high,
# [5] future_state, [6] value

import os
import bisect
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class VivaDataset(Dataset):
    """
    Viva Dataset: predicts future state vector + value only .
    Future state is loaded from frame_index + future_offset.
    """

    def __init__(
        self,
        data_paths: Optional[List[str]] = None,
        video_height: int = 384,
        video_width: int = 320,
        state_stats: Optional[Dict] = None,
        skip_video_decoding: bool = False,
        max_samples: Optional[int] = None,
        future_offset: int = 75,
        task_configs: Optional[List[Dict]] = None,
    ):
        self.video_height = video_height
        self.video_width = video_width
        self.max_samples = max_samples
        self.future_offset = future_offset

        self._subdataset_to_task_id = None
        self.task_configs = task_configs

        if task_configs is not None:
            all_data_paths = []
            self._subdataset_to_task_id = {}
            sub_ds_counter = 0
            for task_idx, tc in enumerate(task_configs):
                for dp in tc['data_paths']:
                    all_data_paths.append(dp)
                    self._subdataset_to_task_id[sub_ds_counter] = task_idx
                    sub_ds_counter += 1
            data_paths = all_data_paths
            logger.info(f"Multi-task mode: {len(task_configs)} tasks, "
                        f"{len(data_paths)} sub-datasets")
        else:
            if data_paths is None:
                raise ValueError("Either data_paths or task_configs must be provided")

        if state_stats is not None:
            self.state_min = torch.tensor(state_stats['state_min'], dtype=torch.float32)
            self.state_max = torch.tensor(state_stats['state_max'], dtype=torch.float32)
        else:
            self.state_min = torch.tensor([-3.0] * 14, dtype=torch.float32)
            self.state_max = torch.tensor([3.0] * 14, dtype=torch.float32)

        from giga_datasets.datasets.dataset import load_dataset

        logger.info("=" * 60)
        logger.info("Loading LeRobot datasets (Viva)...")
        logger.info(f"  Future offset: {future_offset} frames")
        logger.info("=" * 60)

        configs = []
        for input_dir in data_paths:
            configs.append({
                "_class_name": "LeRobotDataset",
                "data_path": str(input_dir),
                "skip_video_decoding": skip_video_decoding,
            })

        self.lerobot_dataset = load_dataset(configs)
        logger.info(f"  Total samples: {len(self.lerobot_dataset)}")

        if not hasattr(self.lerobot_dataset, 'cumulative_sizes'):
            cumsum = []
            total = 0
            for ds in self.lerobot_dataset.datasets:
                total += len(ds)
                cumsum.append(total)
            self.lerobot_dataset.cumulative_sizes = cumsum

        self._subdataset_episode_lengths = []
        max_episode_length = 0
        for dataset in self.lerobot_dataset.datasets:
            sub_lengths = []
            meta = dataset.dataset.meta
            total_episodes = meta.info["total_episodes"]
            for episode_idx in range(total_episodes):
                length = int(meta.episodes[episode_idx]['length'])
                sub_lengths.append(length)
                max_episode_length = max(max_episode_length, length)
            self._subdataset_episode_lengths.append(sub_lengths)

        self.max_episode_length = max_episode_length
        logger.info(f"  Max episode length: {max_episode_length}")
        logger.info(f"  Video size: {video_height}x{video_width}")
        logger.info(f"  State range: [{self.state_min.min():.2f}, {self.state_max.max():.2f}]")

        self._total_samples = len(self.lerobot_dataset)
        self._sample_indices = None
        if self.max_samples is not None and self.max_samples > 0:
            self._effective_length = min(self.max_samples, self._total_samples)
            if self._effective_length < self._total_samples:
                rng = np.random.RandomState(42)
                self._sample_indices = rng.choice(
                    self._total_samples, size=self._effective_length, replace=False)
        else:
            self._effective_length = self._total_samples

    def _get_subdataset_index(self, global_idx: int) -> int:
        return bisect.bisect_right(self.lerobot_dataset.cumulative_sizes, global_idx)

    def __len__(self) -> int:
        return self._effective_length

    def _process_image(self, img) -> torch.Tensor:
        import torch.nn.functional as F
        if isinstance(img, torch.Tensor):
            if img.dim() == 3:
                image = img.float() if img.shape[0] == 3 else img.permute(2, 0, 1).float()
            else:
                image = img.float()
            if image.max() > 1.0:
                image = image / 255.0
        elif hasattr(img, 'convert'):
            img_np = np.array(img.convert('RGB')).astype(np.float32) / 255.0
            image = torch.from_numpy(img_np).permute(2, 0, 1)
        else:
            raise ValueError(f"Unsupported image type: {type(img)}")

        c, h, w = image.shape
        target_h, target_w = self.video_height, self.video_width
        scale = min(target_h / h, target_w / w)
        new_h, new_w = int(h * scale), int(w * scale)
        image = F.interpolate(image.unsqueeze(0), size=(new_h, new_w),
                              mode='bilinear', align_corners=False).squeeze(0)
        pad_h, pad_w = target_h - new_h, target_w - new_w
        image = F.pad(image, (pad_w // 2, pad_w - pad_w // 2,
                              pad_h // 2, pad_h - pad_h // 2), value=0)
        return image

    def _compute_value(self, frame_idx: int, episode_length: int) -> float:
        if episode_length <= 1:
            return 0.5
        return (episode_length - frame_idx - 1) / (episode_length - 1)

    def _normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        state = torch.clamp(state, self.state_min, self.state_max)
        state_normalized = (state - self.state_min) / (self.state_max - self.state_min + 1e-8)
        return state_normalized * 2 - 1

    def _load_state(self, sample) -> torch.Tensor:
        state_raw = sample.get('observation.state', None)
        if state_raw is not None:
            if isinstance(state_raw, torch.Tensor):
                return self._normalize_state(state_raw.float())
            return self._normalize_state(torch.tensor(state_raw, dtype=torch.float32))
        return torch.zeros(14)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        try:
            actual_idx = idx
            if self._sample_indices is not None:
                actual_idx = int(self._sample_indices[idx])

            sample = self.lerobot_dataset[actual_idx]

            sub_ds_idx = self._get_subdataset_index(actual_idx)
            frame_index = sample['frame_index'].item() if isinstance(sample['frame_index'], torch.Tensor) else sample['frame_index']
            episode_index = sample['episode_index'].item() if isinstance(sample['episode_index'], torch.Tensor) else sample['episode_index']
            episode_length = self._subdataset_episode_lengths[sub_ds_idx][episode_index]

            state_normalized = self._load_state(sample)

            cam_high_raw = sample.get('observation.images.cam_high', None)
            cam_left_raw = sample.get('observation.images.cam_left_wrist', None)
            cam_right_raw = sample.get('observation.images.cam_right_wrist', None)

            cam_high = self._process_image(cam_high_raw) if cam_high_raw is not None else torch.zeros(3, self.video_height, self.video_width)
            cam_left_wrist = self._process_image(cam_left_raw) if cam_left_raw is not None else torch.zeros(3, self.video_height, self.video_width)
            cam_right_wrist = self._process_image(cam_right_raw) if cam_right_raw is not None else torch.zeros(3, self.video_height, self.video_width)

            future_frame_idx = min(frame_index + self.future_offset, episode_length - 1)
            delta = future_frame_idx - frame_index

            if delta > 0:
                future_actual_idx = actual_idx + delta
                future_sample = self.lerobot_dataset[future_actual_idx]
                future_state_normalized = self._load_state(future_sample)
            else:
                future_state_normalized = state_normalized.clone()

            value = self._compute_value(frame_index, episode_length)
            value_normalized = value * 2 - 1
            task_id = self._subdataset_to_task_id[sub_ds_idx] if self._subdataset_to_task_id else 0

            return {
                'cam_high': cam_high,
                'cam_left_wrist': cam_left_wrist,
                'cam_right_wrist': cam_right_wrist,
                'state': state_normalized,
                'future_state': future_state_normalized,
                'value': torch.tensor(value, dtype=torch.float32),
                'value_normalized': torch.tensor(value_normalized, dtype=torch.float32),
                'frame_idx': frame_index,
                'episode_length': episode_length,
                'future_frame_idx': future_frame_idx,
                'task_id': task_id,
            }

        except Exception as e:
            logger.warning(f"Error loading sample idx={idx}: {e}")
            dummy_img = torch.zeros(3, self.video_height, self.video_width)
            return {
                'cam_high': dummy_img.clone(), 'cam_left_wrist': dummy_img.clone(),
                'cam_right_wrist': dummy_img.clone(), 'state': torch.zeros(14),
                'future_state': torch.zeros(14),
                'value': torch.tensor(0.5), 'value_normalized': torch.tensor(0.0),
                'frame_idx': 0, 'episode_length': 1, 'future_frame_idx': 0, 'task_id': 0,
            }


def viva_collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return {
        'cam_high': torch.stack([b['cam_high'] for b in batch]),
        'cam_left_wrist': torch.stack([b['cam_left_wrist'] for b in batch]),
        'cam_right_wrist': torch.stack([b['cam_right_wrist'] for b in batch]),
        'state': torch.stack([b['state'] for b in batch]),
        'future_state': torch.stack([b['future_state'] for b in batch]),
        'value': torch.stack([b['value'] for b in batch]),
        'value_normalized': torch.stack([b['value_normalized'] for b in batch]),
        'frame_idx': torch.tensor([b['frame_idx'] for b in batch]),
        'episode_length': torch.tensor([b['episode_length'] for b in batch]),
        'future_frame_idx': torch.tensor([b['future_frame_idx'] for b in batch]),
        'task_id': torch.tensor([b['task_id'] for b in batch], dtype=torch.long),
    }


def compute_dataset_state_stats(data_paths):
    """Compute state min/max statistics across all datasets."""
    from giga_datasets.datasets.dataset import load_dataset

    configs = [{"_class_name": "LeRobotDataset", "data_path": str(p),
                "skip_video_decoding": True} for p in data_paths]
    dataset = load_dataset(configs)

    all_states = []
    for i in range(len(dataset)):
        sample = dataset[i]
        state = sample.get('observation.state', None)
        if state is not None:
            if isinstance(state, torch.Tensor):
                all_states.append(state.float().numpy())
            else:
                all_states.append(np.array(state, dtype=np.float32))

    if not all_states:
        return {'state_min': [-3.0] * 14, 'state_max': [3.0] * 14}

    all_states = np.stack(all_states, axis=0)
    return {
        'state_min': all_states.min(axis=0).tolist(),
        'state_max': all_states.max(axis=0).tolist(),
    }

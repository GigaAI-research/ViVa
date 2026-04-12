# Latent Injection/Extraction utilities for VivaModel (future state + value)
# 7-frame structure: condition frames + future_state + value (no future camera images)
# Sequence: [blank, state, cam_left, cam_right, cam_high, future_state, value]

import torch
import torch.nn as nn
from typing import Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)

# ===== 7-Frame Latent Sequence Indices =====
# Condition frames (mask=1)
BLANK_IDX = 0
STATE_IDX = 1
CAM_LEFT_WRIST_IDX = 2
CAM_RIGHT_WRIST_IDX = 3
CAM_HIGH_IDX = 4

# Target frames (mask=0)
FUTURE_STATE_IDX = 5
VALUE_IDX = 6

NUM_LATENT_FRAMES = 7
NUM_CONDITION_FRAMES = 5
NUM_TARGET_FRAMES = 2

CONDITION_INDICES = [BLANK_IDX, STATE_IDX, CAM_LEFT_WRIST_IDX, CAM_RIGHT_WRIST_IDX, CAM_HIGH_IDX]
TARGET_INDICES = [FUTURE_STATE_IDX, VALUE_IDX]
FUTURE_IMAGE_INDICES = []  # No future image prediction in viva setup

DEFAULT_FUTURE_OFFSET = 50


def replace_latent_with_state(x0, state, state_index=STATE_IDX):
    B = x0.shape[0]
    C, H, W = x0.shape[1], x0.shape[3], x0.shape[4]
    flat = state.reshape(B, -1)
    sd = flat.shape[1]
    le = C * H * W
    nr = (le + sd - 1) // sd
    repeated = flat.repeat(1, nr)[:, :le]
    x0[:, :, state_index, :, :] = repeated.reshape(B, C, H, W)
    return x0


def replace_latent_with_value(x0, value, value_index=VALUE_IDX):
    B = x0.shape[0]
    C, H, W = x0.shape[1], x0.shape[3], x0.shape[4]
    if value.dim() == 1:
        value = value.unsqueeze(1)
    x0[:, :, value_index, :, :] = value.reshape(B, 1, 1, 1).expand(B, C, H, W).to(x0.dtype)
    return x0


def extract_state_from_latent(output_latent, state_dim=14, state_index=STATE_IDX):
    B = output_latent.shape[0]
    flat = output_latent[:, :, state_index, :, :].reshape(B, -1)
    num_copies = flat.shape[1] // state_dim
    copies = flat[:, :num_copies * state_dim].reshape(B, num_copies, state_dim)
    return copies.mean(dim=1)


def extract_value_from_latent(output_latent, value_index=VALUE_IDX):
    return output_latent[:, :, value_index, :, :].mean(dim=(1, 2, 3))


def get_condition_mask(batch_size, num_frames=NUM_LATENT_FRAMES, device=None, dtype=torch.float32):
    """Viva condition mask: [1,1,1,1,1, 0,0]"""
    mask = torch.ones(num_frames, dtype=dtype, device=device)
    for idx in TARGET_INDICES:
        if idx < num_frames:
            mask[idx] = 0
    return mask.reshape(1, 1, num_frames, 1, 1).expand(batch_size, 1, num_frames, 1, 1)


def apply_condition_mask(clean_latent, noisy_latent, condition_mask):
    return clean_latent * condition_mask + noisy_latent * (1 - condition_mask)

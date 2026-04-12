import torch
import torch.nn as nn
import logging
from typing import Dict, Optional
from pathlib import Path

from wan_model import WanVideoModel
from wan.utils.fm import FlowMatchScheduler
from wan.modules.model import sinusoidal_embedding_1d
from viva_utils import (
    replace_latent_with_state, replace_latent_with_value,
    extract_value_from_latent, extract_state_from_latent,
    get_condition_mask, apply_condition_mask,
    BLANK_IDX, STATE_IDX, CAM_LEFT_WRIST_IDX,
    CAM_RIGHT_WRIST_IDX, CAM_HIGH_IDX,
    FUTURE_STATE_IDX, VALUE_IDX,
    FUTURE_IMAGE_INDICES, TARGET_INDICES, NUM_LATENT_FRAMES,
)

logger = logging.getLogger(__name__)


class VivaModel(nn.Module):
    """
    7-frame latent sequence:
    [blank, state, cam_left, cam_right, cam_high, future_state, value]

    Condition mask: [1, 1, 1, 1, 1, 0, 0]
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dtype = torch.bfloat16

        self.video_model = WanVideoModel.from_pretrained(
            checkpoint_path=config.model.wan.checkpoint_path,
            vae_path=config.model.wan.vae_path,
            config_path=config.model.wan.config_path,
            precision=config.model.wan.precision
        )

        logger.info("VivaModel initialized")
        self.device = next(self.video_model.parameters()).device

        self.num_latent_frames = NUM_LATENT_FRAMES  # 7
        self.latent_h = config.common.video_height // 16
        self.latent_w = config.common.video_width // 16
        self.latent_c = 48

        self.loss_weight_future_state = getattr(
            config.training, 'loss_weight_future_state', 1.0)
        self.loss_weight_value = getattr(
            config.training, 'loss_weight_value', 1.0)
        if not hasattr(config.training, 'loss_weight_value') and hasattr(config.training, 'loss_weight_reward'):
            self.loss_weight_value = getattr(config.training, 'loss_weight_reward')

        logger.info(f"  Latent: [{self.latent_c}, {self.num_latent_frames}, "
                     f"{self.latent_h}, {self.latent_w}]")
        logger.info(f"  Loss weights: fut_state={self.loss_weight_future_state}"
                     f", value={self.loss_weight_value}")

        self.fm_scheduler = FlowMatchScheduler(
            shift=5.0, sigma_min=0.0,
            extra_one_step=True, num_train_timesteps=1000
        )
        self.fm_scheduler.set_timesteps(num_inference_steps=1000, training=True)

        total_p = sum(p.numel() for p in self.parameters())
        train_p = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"  Params: {total_p/1e9:.2f}B total, {train_p/1e9:.2f}B trainable")

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        images = images.to(dtype=self.dtype)
        images_normalized = images * 2.0 - 1.0
        images_normalized = images_normalized.unsqueeze(2)
        with torch.no_grad():
            latents = self.video_model.encode_video(images_normalized)
        return latents.to(dtype=self.dtype)

    def _inject_state_to_latent(self, latent, state):
        B = latent.shape[0]
        C, H, W = latent.shape[1], latent.shape[3], latent.shape[4]
        state = state.to(dtype=self.dtype)
        flat = state.flatten(start_dim=1)
        sd = flat.shape[1]
        le = C * H * W
        nr = (le + sd - 1) // sd
        repeated = flat.repeat(1, nr)[:, :le]
        latent[:, :, 0, :, :] = repeated.reshape(B, C, H, W)
        return latent

    def _inject_value_to_latent(self, latent, value):
        B = latent.shape[0]
        C, H, W = latent.shape[1], latent.shape[3], latent.shape[4]
        value = value.to(dtype=self.dtype)
        if value.dim() == 1:
            value = value.unsqueeze(1)
        latent[:, :, 0, :, :] = value.reshape(B, 1, 1, 1).expand(B, C, H, W)
        return latent

    def build_latent_sequence(
        self, state, cam_left_wrist, cam_right_wrist, cam_high,
        future_state, value_normalized,
    ) -> torch.Tensor:
        """Build 7-frame latent sequence [B, 48, 7, H', W']."""
        B = state.shape[0]
        device = state.device

        cam_left_lat = self.encode_images(cam_left_wrist)
        cam_right_lat = self.encode_images(cam_right_wrist)
        cam_high_lat = self.encode_images(cam_high)

        def _zeros():
            return torch.zeros(B, self.latent_c, 1, self.latent_h,
                               self.latent_w, dtype=self.dtype, device=device)

        blank_lat = _zeros()
        state_lat = self._inject_state_to_latent(_zeros(), state)
        fut_state_lat = self._inject_state_to_latent(_zeros(), future_state)
        value_lat = self._inject_value_to_latent(_zeros(), value_normalized)

        return torch.cat([
            blank_lat,                                # 0: blank
            state_lat,                                # 1: current state
            cam_left_lat.to(dtype=self.dtype),         # 2: curr cam_left
            cam_right_lat.to(dtype=self.dtype),        # 3: curr cam_right
            cam_high_lat.to(dtype=self.dtype),         # 4: curr cam_high
            fut_state_lat,                             # 5: future state
            value_lat,                                # 6: value
        ], dim=2)

    def get_time_embedding(self, timesteps, seq_len):
        if timesteps.dim() == 1:
            timesteps = timesteps.unsqueeze(1).expand(timesteps.size(0), seq_len)
        with torch.amp.autocast('cuda', dtype=torch.float32):
            bt = timesteps.size(0)
            t_flat = timesteps.flatten()
            freq_dim = self.video_model.wan_model.freq_dim
            dim = 3072
            t_emb = self.video_model.wan_model.time_embedding(
                sinusoidal_embedding_1d(freq_dim, t_flat)
                .unflatten(0, (bt, seq_len)).float()
            )
            t_proj = self.video_model.wan_model.time_projection(t_emb) \
                .unflatten(2, (6, dim))
        return t_emb, t_proj

    def forward_denoise(self, noisy_latent, timesteps, t5_embeddings,
                        condition_mask, clean_latent):
        model_input = apply_condition_mask(
            clean_latent, noisy_latent, condition_mask).to(dtype=self.dtype)

        patched = self.video_model.wan_model.patch_embedding(model_input)
        video_tokens = patched.flatten(2).transpose(1, 2)
        seq_len = video_tokens.shape[1]
        B = patched.shape[0]

        _, _, T_p, H_p, W_p = patched.shape
        grid_sizes = torch.tensor(
            [T_p, H_p, W_p], dtype=torch.long, device=patched.device
        ).unsqueeze(0).expand(B, -1)

        time_emb, time_proj = self.get_time_embedding(timesteps, seq_len)

        text_len = self.video_model.wan_model.text_len
        if t5_embeddings.shape[1] < text_len:
            pad = t5_embeddings.new_zeros(
                t5_embeddings.shape[0],
                text_len - t5_embeddings.shape[1],
                t5_embeddings.shape[2])
            t5_embeddings = torch.cat([t5_embeddings, pad], dim=1)
        elif t5_embeddings.shape[1] > text_len:
            t5_embeddings = t5_embeddings[:, :text_len]

        t5_embeddings = t5_embeddings.to(dtype=self.dtype)
        context = self.video_model.wan_model.text_embedding(t5_embeddings)

        freqs = self.video_model.wan_model.freqs
        if freqs.device != video_tokens.device:
            freqs = freqs.to(video_tokens.device)
        seq_lens = torch.full(
            (B,), seq_len, dtype=torch.long, device=video_tokens.device)

        with torch.autocast(device_type="cuda", dtype=self.dtype):
            for block in self.video_model.wan_model.blocks:
                with torch.amp.autocast('cuda', dtype=torch.float32):
                    modulation = (
                        block.modulation.unsqueeze(0) + time_proj
                    ).chunk(6, dim=2)
                norm_x = (block.norm1(video_tokens).float()
                          * (1 + modulation[1].squeeze(2))
                          + modulation[0].squeeze(2))
                attn_out = block.self_attn(norm_x, seq_lens, grid_sizes, freqs)
                video_tokens = video_tokens + attn_out * modulation[2].squeeze(2)

                cross_out = block.cross_attn(
                    block.norm3(video_tokens), context, None)
                video_tokens = video_tokens + cross_out

                ffn_in = (block.norm2(video_tokens).float()
                          * (1 + modulation[4].squeeze(2))
                          + modulation[3].squeeze(2))
                ffn_out = block.ffn(ffn_in)
                video_tokens = video_tokens + ffn_out * modulation[5].squeeze(2)

            pred = self.video_model.wan_model.head(video_tokens, time_emb)
            pred = self.video_model.wan_model.unpatchify(pred, grid_sizes)
            pred = torch.stack([u for u in pred], dim=0)

        return pred.to(dtype=self.dtype)

    def training_step(
        self, state, cam_left_wrist, cam_right_wrist, cam_high,
        future_state, value_normalized=None, t5_embeddings=None, return_dict=True,
    ) -> Dict[str, torch.Tensor]:
        B = state.shape[0]
        device = state.device
        if B == 0:
            return self._dummy_output(device)

        if value_normalized is None:
            raise ValueError("value_normalized must be provided")

        clean_latent = self.build_latent_sequence(
            state, cam_left_wrist, cam_right_wrist, cam_high,
            future_state, value_normalized)

        condition_mask = get_condition_mask(B, device=device, dtype=self.dtype)

        timestep_ids = torch.randint(
            0, self.fm_scheduler.num_train_timesteps, (B,), device='cpu')
        timesteps = self.fm_scheduler.timesteps[timestep_ids].to(
            dtype=self.dtype, device=device)
        sigma = self.fm_scheduler.sigmas[timestep_ids].to(
            dtype=self.dtype, device=device).view(B, 1, 1, 1, 1)

        noise = torch.randn_like(clean_latent, dtype=self.dtype)
        noisy_latent = clean_latent * (1 - sigma) + noise * sigma
        noisy_latent = apply_condition_mask(
            clean_latent, noisy_latent, condition_mask)

        target = (noise - clean_latent) * (1 - condition_mask)

        pred = self.forward_denoise(
            noisy_latent, timesteps, t5_embeddings,
            condition_mask, clean_latent)
        pred_masked = pred * (1 - condition_mask)

        fs = nn.functional.mse_loss(
            pred_masked[:, :, FUTURE_STATE_IDX].float(),
            target[:, :, FUTURE_STATE_IDX].float())
        value_loss = nn.functional.mse_loss(
            pred_masked[:, :, VALUE_IDX].float(),
            target[:, :, VALUE_IDX].float())

        total_loss = (self.loss_weight_future_state * fs
                      + self.loss_weight_value * value_loss)

        if return_dict:
            return {
                'total_loss': total_loss,
                'future_state_loss': fs,
                'value_loss': value_loss,
                'sigma_mean': sigma.mean().item(),
            }
        return total_loss

    def _dummy_output(self, device):
        z = lambda: torch.zeros([], device=device, requires_grad=True)
        return {
            'total_loss': z(), 'future_state_loss': z(),
            'value_loss': z(), 'sigma_mean': 0.0,
        }

    @torch.no_grad()
    def predict_value(
        self, state, cam_left_wrist, cam_right_wrist, cam_high,
        t5_embeddings, num_inference_steps=50,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict future state + value via iterative denoising.

        Returns:
          'value': [B] in [0, 1]
          'future_state': [B, 14]  (predicted future state, denormalized to [-1, 1])
        """
        B = state.shape[0]
        device = state.device

        dummy_value = torch.zeros(B, device=device)
        dummy_state = torch.zeros_like(state)

        clean_latent = self.build_latent_sequence(
            state, cam_left_wrist, cam_right_wrist, cam_high,
            dummy_state, dummy_value)

        condition_mask = get_condition_mask(B, device=device, dtype=self.dtype)

        latent = clean_latent.clone()
        for idx in TARGET_INDICES:
            latent[:, :, idx:idx+1] = torch.randn_like(latent[:, :, idx:idx+1])

        self.fm_scheduler.set_timesteps(num_inference_steps)
        sigmas = self.fm_scheduler.sigmas
        for i, t in enumerate(self.fm_scheduler.timesteps):
            ts = torch.full((B,), t, device=device, dtype=self.dtype)
            pred = self.forward_denoise(
                latent, ts, t5_embeddings, condition_mask, clean_latent)
            dt = sigmas[i] - sigmas[i + 1] if i + 1 < len(sigmas) else sigmas[i]
            for idx in TARGET_INDICES:
                latent[:, :, idx] = (
                    latent[:, :, idx]
                    - dt.to(device=device, dtype=self.dtype) * pred[:, :, idx])

        value_norm = extract_value_from_latent(latent)
        value = torch.clamp((value_norm + 1) / 2, 0, 1)
        predicted_future_state = extract_state_from_latent(
            latent, state_dim=14, state_index=FUTURE_STATE_IDX)

        return {
            'value': value,
            'future_state': predicted_future_state,
        }

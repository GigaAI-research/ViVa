#!/usr/bin/env python
# Training script for VivaModel
# Viva: 7-frame structure predicting future state + value (no future camera images)

import os

import argparse
import logging
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from accelerate import Accelerator
from tqdm import tqdm
from accelerate.utils import DeepSpeedPlugin, ProjectConfiguration
import yaml
from omegaconf import OmegaConf

from viva_model import VivaModel
from viva_dataset import (
    VivaDataset, viva_collate_fn,
    compute_dataset_state_stats
)
from utils.scheduler import create_scheduler

logger = logging.getLogger(__name__)


def setup_logging(rank: int = 0, log_level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=f'[Rank {rank}] %(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    warnings.filterwarnings(
        "ignore",
        message=r"No device id is provided via `init_process_group` or `barrier`.*",
        category=UserWarning,
    )


def load_config(config_path: str) -> OmegaConf:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    config = OmegaConf.load(config_path)
    logger.info(f"Loaded config from {config_path}")
    logger.info(f"Video size: {config.common.video_height}x{config.common.video_width}")
    logger.info(f"Num latent frames: {config.common.num_latent_frames}")
    return config


class VivaTrainer:
    """Trainer for VivaModel (future state + value only)."""

    def __init__(
        self, model, train_dataloader, optimizer,
        scheduler=None, device="cuda", rank=0, world_size=1,
        checkpoint_dir="./checkpoints", log_interval=100,
        save_interval=1000, tb_writer=None, accelerator=None,
        config=None, t5_embedding_path=None,
        t5_embedding_paths=None,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.dtype = torch.bfloat16
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.tb_writer = tb_writer
        self.accelerator = accelerator
        self.config = config

        # Load T5 embeddings
        # Multi-task mode: list of embeddings indexed by task_id
        # Single-task mode: single embedding (backward compatible)
        self.t5_embeddings = None  # List[Tensor] for multi-task
        self.t5_embedding = None   # Single Tensor for single-task (backward compat)

        if t5_embedding_paths and len(t5_embedding_paths) > 0:
            # Multi-task mode: load one embedding per task
            self.t5_embeddings = []
            for i, path in enumerate(t5_embedding_paths):
                if path and os.path.exists(path):
                    emb = torch.load(path, map_location='cpu')
                    self.t5_embeddings.append(emb)
                    logger.info(f"Loaded T5 embedding [{i}]: {path}, shape={emb.shape}")
                else:
                    logger.warning(f"T5 embedding [{i}] not found: {path}, using zeros")
                    self.t5_embeddings.append(None)
            logger.info(f"Multi-task T5 embeddings: {len(self.t5_embeddings)} loaded")
        elif t5_embedding_path and os.path.exists(t5_embedding_path):
            # Single-task backward-compatible mode
            self.t5_embedding = torch.load(t5_embedding_path, map_location='cpu')
            logger.info(f"Loaded T5 embedding (single): {self.t5_embedding.shape}")

        if rank == 0:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.global_step = 0
        self.epoch = 0
        logger.info(f"VivaTrainer initialized on rank {rank}/{world_size}")

    def save_checkpoint(self, suffix: str = ""):
        checkpoint_dir = self.checkpoint_dir / f"checkpoint_step_{self.global_step}{suffix}"
        self.accelerator.save_state(str(checkpoint_dir))
        logger.info(f"Checkpoint saved to {checkpoint_dir}")
        try:
            cfg_dict = OmegaConf.to_container(self.config, resolve=True) if self.config else {}
            import json
            with open(checkpoint_dir / "config.json", "w") as f:
                json.dump(cfg_dict, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save config: {e}")

    def load_checkpoint(self, checkpoint_path: str):
        if not os.path.exists(checkpoint_path):
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
            return
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        import re
        step_match = re.search(r'step_(\d+)', checkpoint_path)
        if step_match:
            self.global_step = int(step_match.group(1))
            logger.info(f"Resuming from step {self.global_step}")
        self.accelerator.load_state(checkpoint_path)

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step with future state + value."""
        self.model.train()

        # Current observations
        state = batch['state'].to(self.device, dtype=self.dtype)
        cam_high = batch['cam_high'].to(self.device, dtype=self.dtype)
        cam_left = batch['cam_left_wrist'].to(self.device, dtype=self.dtype)
        cam_right = batch['cam_right_wrist'].to(self.device, dtype=self.dtype)

        # Future state
        fut_state = batch['future_state'].to(self.device, dtype=self.dtype)

        # Value
        value_normalized = batch['value_normalized'].to(self.device, dtype=self.dtype)

        # T5 embedding (multi-task or single-task)
        B = state.shape[0]
        if self.t5_embeddings is not None:
            # Multi-task: select per-sample T5 embedding by task_id
            task_ids = batch['task_id']  # [B] LongTensor
            emb_list = []
            max_seq_len = 0
            for i in range(B):
                tid = task_ids[i].item()
                emb = self.t5_embeddings[tid]
                if emb is None:
                    emb = torch.zeros(4, 4096)
                emb_list.append(emb)
                max_seq_len = max(max_seq_len, emb.shape[0])
            # Pad to same seq_len and stack
            padded = []
            for emb in emb_list:
                if emb.shape[0] < max_seq_len:
                    pad = torch.zeros(max_seq_len - emb.shape[0], emb.shape[1],
                                      dtype=emb.dtype)
                    emb = torch.cat([emb, pad], dim=0)
                padded.append(emb)
            t5_emb = torch.stack(padded, dim=0)  # [B, max_seq_len, 4096]
            t5_emb = t5_emb.to(self.device, dtype=self.dtype)
        elif self.t5_embedding is not None:
            # Single-task backward compatible
            t5_emb = self.t5_embedding.unsqueeze(0).expand(B, -1, -1)
            t5_emb = t5_emb.to(self.device, dtype=self.dtype)
        else:
            t5_emb = torch.zeros(B, 4, 4096, device=self.device, dtype=self.dtype)

        model = self.model.module if hasattr(self.model, 'module') else self.model

        with self.accelerator.accumulate(model):
            loss_dict = model.training_step(
                state=state,
                cam_left_wrist=cam_left,
                cam_right_wrist=cam_right,
                cam_high=cam_high,
                future_state=fut_state,
                value_normalized=value_normalized,
                t5_embeddings=t5_emb,
                return_dict=True
            )

            total_loss = loss_dict['total_loss']
            self.accelerator.backward(total_loss)

            if self.accelerator.sync_gradients:
                grad_clip = getattr(self.config.training, 'grad_clip_norm', 1.0)
                self.accelerator.clip_grad_norm_(self.model.parameters(), grad_clip)
                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()
                self.optimizer.zero_grad()

        metrics = {k: v.item() if torch.is_tensor(v) else v
                   for k, v in loss_dict.items()}
        return metrics

    def train(self, max_steps=None, max_epochs=None,
              resume_from=None, reset_step=False):
        if resume_from:
            self.load_checkpoint(resume_from)
            if reset_step:
                logger.info("Resetting global_step to 0 (fine-tune mode)")
                self.global_step = 0

        use_epoch_mode = max_steps is None and max_epochs is not None

        if hasattr(self.train_dataloader, 'dataset'):
            dataset_size = len(self.train_dataloader.dataset)
        else:
            dataset_size = len(self.train_dataloader) * self.config.training.batch_size * self.world_size

        batch_size = self.config.training.batch_size
        batches_per_epoch = dataset_size // (batch_size * self.world_size)
        logger.info(f"Dataset: {dataset_size}, batches/epoch: {batches_per_epoch}")

        if use_epoch_mode:
            total_batches = max_epochs * batches_per_epoch
            logger.info(f"Training for {max_epochs} epochs ({total_batches} batches)")
        else:
            total_batches = max_steps if max_steps else 1000
            logger.info(f"Training for {total_batches} batches")

        start_time = time.time()
        epoch = self.global_step // batches_per_epoch if batches_per_epoch > 0 else 0
        step_in_epoch = self.global_step % batches_per_epoch if batches_per_epoch > 0 else 0

        if hasattr(self.train_dataloader.sampler, 'set_epoch'):
            self.train_dataloader.sampler.set_epoch(epoch)

        if step_in_epoch > 0:
            logger.info(f"Fast-skipping {step_in_epoch} batches via accelerate...")
            from accelerate import skip_first_batches
            skipped_dataloader = skip_first_batches(self.train_dataloader, step_in_epoch)
            data_iter = iter(skipped_dataloader)
            logger.info(f"Skip done, resuming training from step {self.global_step}")
        else:
            data_iter = iter(self.train_dataloader)

        pbar = tqdm(
            initial=self.global_step, total=total_batches,
            desc="Training Viva", disable=(self.rank != 0),
            dynamic_ncols=True, smoothing=0.1,
        )

        running_total_loss = 0.0
        running_state_loss = 0.0
        running_value_loss = 0.0
        loss_count = 0
        steps_this_epoch = step_in_epoch

        while self.global_step < total_batches:
            # Proactive epoch switch: all ranks switch together based on step count
            if steps_this_epoch >= batches_per_epoch:
                epoch += 1
                steps_this_epoch = 0
                logger.info(f"Epoch {epoch-1} done, starting epoch {epoch}")
                if self.global_step > 0:
                    self.save_checkpoint(suffix=f"_epoch{epoch-1}")
                if hasattr(self.train_dataloader.sampler, 'set_epoch'):
                    self.train_dataloader.sampler.set_epoch(epoch)
                data_iter = iter(self.train_dataloader)

            try:
                batch = next(data_iter)
            except StopIteration:
                logger.warning(f"DataLoader exhausted at step_in_epoch={steps_this_epoch}, "
                               f"expected {batches_per_epoch}. Re-creating iterator.")
                data_iter = iter(self.train_dataloader)
                batch = next(data_iter)

            if batch is None:
                steps_this_epoch += 1
                continue

            step_start = time.time()
            metrics = self.train_step(batch)
            step_time = time.time() - step_start
            self.global_step += 1
            steps_this_epoch += 1

            running_total_loss += metrics['total_loss']
            running_state_loss += metrics['future_state_loss']
            running_value_loss += metrics['value_loss']
            loss_count += 1

            lr = self.optimizer.param_groups[0]['lr']
            current_epoch = self.global_step // batches_per_epoch if batches_per_epoch > 0 else 0

            if self.rank == 0:
                pbar.set_postfix({
                    'total': f"{metrics['total_loss']:.4f}",
                    'fut_st': f"{metrics['future_state_loss']:.4f}",
                    'value': f"{metrics['value_loss']:.4f}",
                    'lr': f"{lr:.2e}",
                    'ep': current_epoch,
                }, refresh=False)
                pbar.update(1)

            if self.global_step % self.log_interval == 0 and self.rank == 0:
                if self.tb_writer:
                    for k, v in metrics.items():
                        self.tb_writer.add_scalar(f'train/{k}', v, self.global_step)
                    self.tb_writer.add_scalar('train/learning_rate', lr, self.global_step)
                    self.tb_writer.add_scalar('train/step_time', step_time, self.global_step)
                    self.tb_writer.add_scalar('train/avg_total_loss',
                        running_total_loss / loss_count, self.global_step)
                    self.tb_writer.add_scalar('train/avg_future_state_loss',
                        running_state_loss / loss_count, self.global_step)
                    self.tb_writer.add_scalar('train/avg_value_loss',
                        running_value_loss / loss_count, self.global_step)
                    self.tb_writer.add_scalar('train/epoch', current_epoch, self.global_step)

            if self.global_step % self.save_interval == 0:
                self.save_checkpoint()

        pbar.close()
        total_time = time.time() - start_time
        if self.rank == 0:
            logger.info(f"Training completed in {total_time:.2f}s "
                        f"({self.global_step} steps, {epoch} epochs)")
            self.save_checkpoint()


def create_model_and_optimizer(config):
    model = VivaModel(config)

    base_lr = float(config.training.learning_rate)
    wan_lr = float(getattr(config.training, 'wan_learning_rate', base_lr))

    wan_params = [p for p in model.video_model.wan_model.parameters() if p.requires_grad]
    all_trainable = [p for p in model.parameters() if p.requires_grad]
    wan_ids = {id(p) for p in wan_params}
    other_params = [p for p in all_trainable if id(p) not in wan_ids]

    param_groups = []
    if other_params:
        param_groups.append({'params': other_params, 'lr': base_lr})
    if wan_params:
        param_groups.append({'params': wan_params, 'lr': wan_lr})

    optimizer = torch.optim.AdamW(
        param_groups, weight_decay=config.training.weight_decay, betas=(0.9, 0.95))
    scheduler = create_scheduler(optimizer, config)

    return model, optimizer, scheduler


def create_dataloaders(config, rank, world_size):
    # Determine if multi-task mode (config.dataset.tasks) or single-task (config.dataset.data_paths)
    use_multi_task = hasattr(config.dataset, 'tasks') and config.dataset.tasks is not None
    task_configs = None

    if use_multi_task:
        task_configs = []
        all_data_paths = []
        for tc in config.dataset.tasks:
            tc_dict = {
                'task_id': tc.get('task_id', 'unknown'),
                't5_embedding_path': tc.get('t5_embedding_path', ''),
                'data_paths': list(tc.data_paths),
            }
            task_configs.append(tc_dict)
            all_data_paths.extend(tc_dict['data_paths'])
        data_paths = all_data_paths
        logger.info(f"Multi-task mode: {len(task_configs)} tasks, {len(data_paths)} total datasets")
    else:
        data_paths = config.dataset.data_paths
        if isinstance(data_paths, str):
            data_paths = [data_paths]

    logger.info("Computing state statistics...")
    state_stats = None
    if hasattr(config.dataset, 'compute_state_stats') and config.dataset.compute_state_stats:
        state_stats = compute_dataset_state_stats(data_paths)
        logger.info(f"State min: {state_stats['state_min']}")
        logger.info(f"State max: {state_stats['state_max']}")

    future_offset = getattr(config.dataset, 'future_offset', 75)
    max_samples = getattr(config.dataset, 'max_samples', None)

    if use_multi_task:
        train_dataset = VivaDataset(
            video_height=config.common.video_height,
            video_width=config.common.video_width,
            state_stats=state_stats,
            max_samples=max_samples,
            future_offset=future_offset,
            task_configs=task_configs,
        )
    else:
        train_dataset = VivaDataset(
            data_paths=data_paths,
            video_height=config.common.video_height,
            video_width=config.common.video_width,
            state_stats=state_stats,
            max_samples=max_samples,
            future_offset=future_offset,
        )

    logger.info(f"Viva dataset size: {len(train_dataset)}, future_offset: {future_offset}")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.system.num_workers,
        pin_memory=config.system.pin_memory,
        collate_fn=viva_collate_fn,
        drop_last=True,
    )

    return train_dataloader, task_configs


def main():
    parser = argparse.ArgumentParser(description="Train VivaModel")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--log_level", type=str, default="INFO")
    parser.add_argument("--deepspeed", type=str, default=None)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--loss_weight_future_state", type=float, default=None,
                        help="Override config.training.loss_weight_future_state")
    args = parser.parse_args()

    config = load_config(args.config)

    # Override loss weight from command line
    if args.loss_weight_future_state is not None:
        config.training.loss_weight_future_state = args.loss_weight_future_state
        logger.info(f"CLI override: loss_weight_future_state = {args.loss_weight_future_state}")

    base_checkpoint_dir = args.checkpoint_dir if args.checkpoint_dir else config.system.checkpoint_dir
    if args.run_name:
        config.system.checkpoint_dir = os.path.join(base_checkpoint_dir, args.run_name)
    elif args.checkpoint_dir:
        # If user explicitly passes --checkpoint_dir, use it directly without auto suffixes.
        config.system.checkpoint_dir = base_checkpoint_dir
    else:
        default_run_name = getattr(config.logging, "run_name", None)
        if not default_run_name:
            default_run_name = os.path.splitext(os.path.basename(args.config))[0]
        config.system.checkpoint_dir = os.path.join(base_checkpoint_dir, default_run_name)
    os.makedirs(config.system.checkpoint_dir, exist_ok=True)

    accelerator_config = ProjectConfiguration(total_limit=20)
    accelerator = Accelerator(
        deepspeed_plugin=DeepSpeedPlugin(hf_ds_config=args.deepspeed) if args.deepspeed else None,
        gradient_accumulation_steps=config.training.get('gradient_accumulation_steps', 1),
        mixed_precision="bf16",
        log_with='tensorboard',
        project_dir=config.system.checkpoint_dir,
        project_config=accelerator_config,
    )

    rank = accelerator.process_index
    world_size = accelerator.num_processes
    setup_logging(rank, args.log_level)

    logger.info(f"Checkpoints: {config.system.checkpoint_dir}")

    tb_writer = None
    if rank == 0:
        tb_log_dir = os.path.join(config.system.checkpoint_dir,
                                   config.logging.tensorboard_log_dir)
        tb_writer = SummaryWriter(log_dir=tb_log_dir)
        logger.info(f"TensorBoard: {tb_log_dir}")
        config_dict = OmegaConf.to_container(config, resolve=True)
        tb_writer.add_text('config', yaml.dump(config_dict))

    try:
        logger.info("Creating Viva model...")
        model, optimizer, scheduler = create_model_and_optimizer(config)

        logger.info("Creating Viva dataloaders...")
        train_dataloader, task_configs = create_dataloaders(config, rank, world_size)

        logger.info("Preparing with Accelerator...")
        model, optimizer, train_dataloader, scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, scheduler)

        # Determine T5 embedding paths
        t5_embedding_paths = None
        t5_path = None

        if task_configs is not None:
            t5_embedding_paths = [tc['t5_embedding_path'] for tc in task_configs]
            logger.info(f"Multi-task T5 embeddings: {t5_embedding_paths}")
        else:
            t5_path = getattr(config.dataset, 't5_embedding_path', None)
            if not t5_path:
                t5_path = "/shared_disk/users/hao.li/policy/data/t5_stack_box.pt"

        trainer = VivaTrainer(
            model=model,
            train_dataloader=train_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=accelerator.device,
            rank=rank,
            world_size=world_size,
            checkpoint_dir=config.system.checkpoint_dir,
            log_interval=config.system.log_interval,
            save_interval=config.system.save_interval,
            tb_writer=tb_writer,
            accelerator=accelerator,
            config=config,
            t5_embedding_path=t5_path,
            t5_embedding_paths=t5_embedding_paths,
        )

        resume_from = getattr(config.resume, 'checkpoint_path', None)
        reset_step = getattr(config.resume, 'reset_step', False)
        max_steps = getattr(config.training, 'max_steps', None)
        max_epochs = getattr(config.training, 'max_epochs', None)

        if max_steps is None and max_epochs is None:
            max_epochs = 1
            logger.info("Defaulting to 1 epoch")

        trainer.train(
            max_steps=max_steps,
            max_epochs=max_epochs,
            resume_from=resume_from,
            reset_step=reset_step,
        )

    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        if tb_writer:
            tb_writer.close()


if __name__ == "__main__":
    main()

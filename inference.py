
"""
Viva - Multi-GPU batch inference that writes model predictions back to parquet files.


Features:
1. Iterate over all frames of all episodes under all `data_paths`.
2. Split work across N GPUs, running per-frame inference on each GPU.
3. Generate two new fields for each frame:
   - prediction: raw model output value in [0, 1]
   - model_prediction: denormalized to frame-index scale, i.e. predicted remaining frames
     Formula: predicted_remaining = prediction * (ep_len - 1)
4. Append `prediction` and `model_prediction` to the original parquet file.

Usage:
  # Launch with torchrun for multi-GPU (recommended)
  torchrun --nproc_per_node=8 inference.py 

  # Or use mp.spawn (automatic)
  python inference.py --num_gpus 8 

  # Single-GPU debugging
  python inference.py --num_gpus 1 

Note: --config is required because it controls whether to compute training-set state statistics , 
which are needed for min-max normalization of the state vectors.
"""

import os
import argparse
import ast
import logging
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from omegaconf import OmegaConf

from viva_model import VivaModel
from viva_dataset import VivaDataset, compute_dataset_state_stats

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# 1. Model Loading

def load_model(checkpoint_dir: str, config, device: str = "cuda"):
    """Load VivaModel from an Accelerate checkpoint."""
    from safetensors.torch import load_file

    logger.info(f"Loading VivaModel from {checkpoint_dir}")
    model = VivaModel(config)

    model_file = None
    for name in ['model.safetensors', 'pytorch_model.bin']:
        path = os.path.join(checkpoint_dir, name)
        if os.path.exists(path):
            model_file = path
            break

    if model_file is None:
        raise FileNotFoundError(f"No model weights found in {checkpoint_dir}")

    logger.info(f"Loading weights from {model_file}")
    if model_file.endswith('.safetensors'):
        state_dict = load_file(model_file)
    else:
        state_dict = torch.load(model_file, map_location='cpu')

    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    logger.info("Model loaded successfully")
    return model


# 2. Build Episode Metadata

def build_episode_table(dataset: VivaDataset) -> List[Dict[str, Any]]:
    """
    Build an episode metadata table:
    [{sub_ds_idx, episode_index, episode_length, global_start, task}]
    """
    table = []
    global_offset = 0
    for sub_ds_idx, sub_lengths in enumerate(dataset._subdataset_episode_lengths):
        sub_ds = dataset.lerobot_dataset.datasets[sub_ds_idx]
        meta = sub_ds.dataset.meta
        tasks = meta.tasks
        task_name = tasks.get(0, "unknown") if isinstance(tasks, dict) else str(tasks)

        for ep_idx, ep_len in enumerate(sub_lengths):
            table.append({
                "sub_ds_idx": sub_ds_idx,
                "episode_index": ep_idx,
                "episode_length": ep_len,
                "task": task_name,
                "global_start": global_offset,
            })
            global_offset += ep_len

    return table


def get_parquet_path(dataset: VivaDataset, sub_ds_idx: int, episode_index: int) -> Path:
    """Get the parquet file path for a specific episode."""
    sub_ds = dataset.lerobot_dataset.datasets[sub_ds_idx]
    data_path = Path(sub_ds.dataset.root)
    chunk_idx = episode_index // 1000
    parquet_path = (data_path / f"data/chunk-{chunk_idx:03d}"
                    / f"episode_{episode_index:06d}.parquet")
    return parquet_path


def load_state_txt(path: str):
    """Parse state_stats txt file and return {'state_min': array, 'state_max': array}."""
    state_min = None
    state_max = None
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("state_min ="):
                val_str = line.split("=", 1)[1].strip()
                state_min = ast.literal_eval(val_str)
            elif line.startswith("state_max ="):
                val_str = line.split("=", 1)[1].strip()
                state_max = ast.literal_eval(val_str)
    if state_min is None or state_max is None:
        raise ValueError(f"Could not parse state_min/state_max from {path}")
    return {
        "state_min": np.array(state_min, dtype=np.float32),
        "state_max": np.array(state_max, dtype=np.float32),
    }


# 3. Single-Frame Inference

@torch.no_grad()
def infer_single_frame(
    model: VivaModel,
    dataset: VivaDataset,
    global_idx: int,
    ep_info: Dict,
    t5_embedding: Optional[torch.Tensor],
    device: str = "cuda",
    num_inference_steps: int = 1,
    dtype=torch.bfloat16,
) -> Dict[str, float]:
    """
    Run inference for a single frame and return prediction and model_prediction.

    Returns:
        {
            "prediction": float,        # Model-predicted value in [0, 1]
            "model_prediction": float,   # Denormalized remaining frame count
            "frame_index": int,          # Ground-truth frame index
        }
    """
    sample = dataset.lerobot_dataset[global_idx]

    frame_index = sample['frame_index'].item() if isinstance(sample['frame_index'], torch.Tensor) else sample['frame_index']
    episode_length = ep_info["episode_length"]

    # state
    state_raw = sample.get('observation.state', None)
    if state_raw is not None:
        if isinstance(state_raw, torch.Tensor):
            state = state_raw.float()
        else:
            state = torch.tensor(state_raw, dtype=torch.float32)
        state = torch.clamp(state, dataset.state_min, dataset.state_max)
        state = (state - dataset.state_min) / (dataset.state_max - dataset.state_min + 1e-8)
        state = state * 2 - 1
    else:
        state = torch.zeros(14)

    # images
    cam_high_raw = sample.get('observation.images.cam_high')
    cam_left_raw = sample.get('observation.images.cam_left_wrist')
    cam_right_raw = sample.get('observation.images.cam_right_wrist')

    cam_high = dataset._process_image(cam_high_raw) if cam_high_raw is not None else torch.zeros(3, dataset.video_height, dataset.video_width)
    cam_left = dataset._process_image(cam_left_raw) if cam_left_raw is not None else torch.zeros(3, dataset.video_height, dataset.video_width)
    cam_right = dataset._process_image(cam_right_raw) if cam_right_raw is not None else torch.zeros(3, dataset.video_height, dataset.video_width)

    # batch dim = 1
    state_b = state.unsqueeze(0).to(device=device, dtype=dtype)
    cam_high_b = cam_high.unsqueeze(0).to(device=device, dtype=dtype)
    cam_left_b = cam_left.unsqueeze(0).to(device=device, dtype=dtype)
    cam_right_b = cam_right.unsqueeze(0).to(device=device, dtype=dtype)

    # t5 embedding
    if t5_embedding is not None:
        t5_emb = t5_embedding.unsqueeze(0).to(device=device, dtype=dtype)
    else:
        t5_emb = torch.zeros(1, 4, 4096, device=device, dtype=dtype)

    # Inference - predict_value returns {'value': [B], 'future_state': [B, 14]}
    output = model.predict_value(
        state_b, cam_left_b, cam_right_b, cam_high_b,
        t5_emb, num_inference_steps=num_inference_steps,
    )
    pred_value = output['value'][0].float().item()

    # Denormalize: prediction -> model_prediction (predicted remaining frames)
    if episode_length <= 1:
        predicted_remaining = 0.0
    else:
        predicted_remaining = pred_value * (episode_length - 1)

    return {
        "prediction": pred_value,
        "model_prediction": predicted_remaining,
        "frame_index": frame_index,
    }


# 4. Infer Full Episode and Write Back to Parquet

@torch.no_grad()
def infer_and_annotate_episode(
    model: VivaModel,
    dataset: VivaDataset,
    ep_info: Dict,
    t5_embedding: Optional[torch.Tensor],
    device: str = "cuda",
    num_inference_steps: int = 1,
    rank: int = 0,
) -> Dict[str, Any]:
    """
    Run frame-by-frame inference for a full episode and write results back to parquet.

    Added fields:
        - prediction: Model-predicted value in [0, 1]
        - model_prediction: Denormalized remaining frame count
    """
    global_start = ep_info["global_start"]
    ep_len = ep_info["episode_length"]
    sub_ds_idx = ep_info["sub_ds_idx"]
    episode_index = ep_info["episode_index"]

    predictions = []
    model_predictions = []

    dtype = torch.bfloat16

    for i in range(ep_len):
        global_idx = global_start + i
        result = infer_single_frame(
            model=model,
            dataset=dataset,
            global_idx=global_idx,
            ep_info=ep_info,
            t5_embedding=t5_embedding,
            device=device,
            num_inference_steps=num_inference_steps,
            dtype=dtype,
        )
        predictions.append(result["prediction"])
        model_predictions.append(result["model_prediction"])

    # Read original parquet file
    parquet_path = get_parquet_path(dataset, sub_ds_idx, episode_index)

    if not parquet_path.exists():
        logger.warning(f"[Rank {rank}] Parquet file not found: {parquet_path}")
        return {
            "sub_ds_idx": sub_ds_idx,
            "episode_index": episode_index,
            "status": "file_not_found",
            "parquet_path": str(parquet_path),
        }

    # Read original table
    table = pq.read_table(str(parquet_path))
    num_rows = table.num_rows

    if num_rows != ep_len:
        logger.warning(
            f"[Rank {rank}] Row count mismatch for ep {episode_index}: "
            f"parquet has {num_rows} rows, but episode_length={ep_len}. "
            f"Using min({num_rows}, {ep_len}) rows."
        )
        min_len = min(num_rows, ep_len)
        predictions = predictions[:min_len]
        model_predictions = model_predictions[:min_len]

    # Add new columns (remove existing ones first to support re-runs)
    existing_cols = table.column_names
    for col_name in ['prediction', 'model_prediction']:
        if col_name in existing_cols:
            col_idx = existing_cols.index(col_name)
            table = table.remove_column(col_idx)
            existing_cols = table.column_names

    table = table.append_column('prediction', pa.array(predictions, type=pa.float64()))
    table = table.append_column('model_prediction', pa.array(model_predictions, type=pa.float64()))

    # Write back to parquet file
    pq.write_table(table, str(parquet_path))

    return {
        "sub_ds_idx": sub_ds_idx,
        "episode_index": episode_index,
        "episode_length": ep_len,
        "status": "success",
        "parquet_path": str(parquet_path),
    }


# 5. Single-GPU Worker Process

def worker_fn(
    rank: int,
    world_size: int,
    args: argparse.Namespace,
    config,
    data_paths: List[str],
    episode_table: List[Dict],
    state_stats: Optional[Dict] = None,
):
    """
    Main function for each GPU worker.
    Episodes are assigned to GPUs in a round-robin manner.
    """
    device = f"cuda:{rank}"
    torch.cuda.set_device(rank)

    logger.info(f"[Rank {rank}/{world_size}] Starting on {device}")

    # Load model
    model = load_model(args.checkpoint, config, device)

    # T5 embedding
    t5_embedding = None
    if args.t5_embedding and os.path.exists(args.t5_embedding):
        t5_embedding = torch.load(args.t5_embedding, map_location='cpu')
        logger.info(f"[Rank {rank}] T5 embedding loaded: {t5_embedding.shape}")

    # Load dataset
    dataset = VivaDataset(
        data_paths=data_paths,
        video_height=config.common.video_height,
        video_width=config.common.video_width,
        state_stats=state_stats,
    )

    # Assign episodes by rank (round-robin)
    my_episodes = [ep for i, ep in enumerate(episode_table) if i % world_size == rank]
    total_frames = sum(ep["episode_length"] for ep in my_episodes)

    logger.info(
        f"[Rank {rank}] Assigned {len(my_episodes)} episodes, "
        f"{total_frames} total frames"
    )

    results = []
    processed_frames = 0

    # Show tqdm progress bar on rank 0 only; keep other ranks silent
    ep_iter = tqdm(
        my_episodes,
        desc=f"[Rank {rank}] episodes",
        position=rank,
        leave=True,
        disable=(rank != 0),
    )

    for ep_info in ep_iter:
        ep_len = ep_info["episode_length"]

        result = infer_and_annotate_episode(
            model=model,
            dataset=dataset,
            ep_info=ep_info,
            t5_embedding=t5_embedding,
            device=device,
            num_inference_steps=args.num_inference_steps,
            rank=rank,
        )
        results.append(result)
        processed_frames += ep_len

        status_str = "OK" if result["status"] == "success" else result["status"]
        ep_iter.set_postfix(
            ep=ep_info["episode_index"],
            frames=f"{processed_frames}/{total_frames}",
            status=status_str,
        )

        if result["status"] != "success":
            logger.warning(
                f"[Rank {rank}] Failed: {result['status']} - {result['parquet_path']}"
            )

    # Save failure summary for current rank (only failures)
    failed_results = [r for r in results if r["status"] != "success"]
    summary_path = os.path.join(
        args.output_dir,
        f"annotate_failures_rank{rank}.json"
    )
    os.makedirs(args.output_dir, exist_ok=True)
    if failed_results:
        with open(summary_path, 'w') as f:
            json.dump({
                "rank": rank,
                "world_size": world_size,
                "num_failed": len(failed_results),
                "checkpoint": args.checkpoint,
                "num_inference_steps": args.num_inference_steps,
                "failed_results": failed_results,
            }, f, indent=2)
        logger.info(f"[Rank {rank}] Completed. Failure summary saved to {summary_path}")
    else:
        logger.info(f"[Rank {rank}] Completed. No failures.")

    return results


# 6. Distributed Entry (torchrun)

def run_distributed(args, config, data_paths, episode_table, state_stats=None):
    """Distributed inference entrypoint launched by torchrun."""
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)

    from datetime import timedelta
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        timeout=timedelta(hours=3)
    )

    worker_fn(rank, world_size, args, config, data_paths, episode_table, state_stats)

    logger.info(f"[Rank {rank}] Waiting for other processes...")
    dist.barrier()

    if rank == 0:
        logger.info("=" * 60)
        logger.info("All GPUs finished. Merging summaries...")
        merge_summaries(args.output_dir, world_size)
        logger.info("=" * 60)

    dist.destroy_process_group()


# 7. Multi-Process Entry (mp.spawn)

def spawn_worker(rank, world_size, args, config, data_paths, episode_table, state_stats=None):
    """Wrapper used by mp.spawn."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    worker_fn(rank, world_size, args, config, data_paths, episode_table, state_stats)

    dist.barrier()
    if rank == 0:
        logger.info("=" * 60)
        logger.info("All GPUs finished. Merging summaries...")
        merge_summaries(args.output_dir, world_size)
        logger.info("=" * 60)

    dist.destroy_process_group()


def run_mp_spawn(args, config, data_paths, episode_table, world_size, state_stats=None):
    """Launch multi-process inference with mp.spawn."""
    mp.spawn(
        spawn_worker,
        args=(world_size, args, config, data_paths, episode_table, state_stats),
        nprocs=world_size,
        join=True,
    )


# 8. Single-GPU Entry

def run_single_gpu(args, config, data_paths, episode_table, state_stats=None):
    """Single-GPU inference."""
    worker_fn(0, 1, args, config, data_paths, episode_table, state_stats)


# 9. Merge Per-Rank Summaries

def merge_summaries(output_dir: str, world_size: int):
    """Merge failure summaries from all ranks and remove per-rank JSON files."""
    all_failures = []
    total_failures = 0

    for rank in range(world_size):
        summary_path = os.path.join(output_dir, f"annotate_failures_rank{rank}.json")
        if os.path.exists(summary_path):
            with open(summary_path, 'r') as f:
                summary = json.load(f)
            failures = summary.get("failed_results", [])
            all_failures.extend(failures)
            total_failures += len(failures)
            os.remove(summary_path)
            logger.info(f"Removed temporary failure summary for rank {rank}: {summary_path}")

    if all_failures:
        merged = {
            "total_failures": total_failures,
            "failed_results": all_failures,
        }
        merged_path = os.path.join(output_dir, "annotate_failures.json")
        with open(merged_path, 'w') as f:
            json.dump(merged, f, indent=2)
        logger.info(f"Failure summary saved: {merged_path}")
        logger.info(f"  Total failures: {total_failures}")
    else:
        logger.info("No failures encountered across all ranks.")


# 10. Main

def main():
    parser = argparse.ArgumentParser(
        description="Viva: Multi-GPU batch inference that writes predictions back to parquet files.")
    parser.add_argument("--checkpoint", type=str,
                        default="./checkpoints/3task/checkpoint_step_45069",
                        help="VivaModel checkpoint directory")
    parser.add_argument("--config", type=str,
                        default="config/train_8gpu_3task.yaml",
                        help="Model config file. Passing it is recommended so that state statistics can be computed (e.g., compute_state_stats) for min-max normalization of state vectors, keeping inference consistent with training.")
    parser.add_argument("--data_path", type=str, nargs='+',
                        default=[
                            "data/toilet_paper_newaction100_3",
                        ],
                        help="List of LeRobot data directories")
    parser.add_argument("--t5_embedding", type=str,
                        default="../data/stack_box.pt",
                        help="Path to T5 embedding")
    parser.add_argument("--num_gpus", type=int, default=8,
                        help="Number of GPUs to use (default: 8)")
    parser.add_argument("--gpus", type=int, nargs='+', default=None,
                        help="Specific GPU ID list (e.g. --gpus 0 1 2 3)")
    parser.add_argument("--num_inference_steps", type=int, default=1,
                        help="Number of denoising steps (default: 1)")
    parser.add_argument("--output_dir", type=str,
                        default="logs/anno",
                        help="Output directory for summary files")
    parser.add_argument("--launch_mode", type=str, default="auto",
                        choices=["auto", "torchrun", "spawn", "single"],
                        help="Launch mode: auto(auto-detect), torchrun, spawn(mp.spawn), single(single GPU)")
    parser.add_argument("--max_episodes_per_dataset", type=int, default=None,
                        help="Only process first N episodes per sub-dataset (for debugging)")
    parser.add_argument("--state_txt", type=str, default=None,
                        help="Path to pre-computed state stats file (e.g., state_stats_3task.txt); will be read directly if provided")
    args = parser.parse_args()

    # Load config
    config = OmegaConf.load(args.config)
    os.makedirs(args.output_dir, exist_ok=True)

    # GPU setup
    if args.gpus:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, args.gpus))
        args.num_gpus = len(args.gpus)

    data_paths = args.data_path

    logger.info("=" * 60)
    logger.info("Viva - Batch Inference & Annotate (Multi-GPU)")
    logger.info("=" * 60)
    logger.info(f"  Checkpoint: {args.checkpoint}")
    logger.info(f"  Config:     {args.config}")
    logger.info(f"  Num GPUs:   {args.num_gpus}")
    logger.info(f"  Num inference steps: {args.num_inference_steps}")
    logger.info(f"  Data paths: {data_paths}")
    logger.info(f"  Output dir: {args.output_dir}")

    # Determine state stats
    state_stats = None
    if args.state_txt and os.path.exists(args.state_txt):
        state_stats = load_state_txt(args.state_txt)
        logger.info(f"Loaded state stats from {args.state_txt}")
    elif config.dataset.get("compute_state_stats", False):
        logger.info("Computing state statistics from data_path (compute_state_stats=true in config)...")
        state_stats = compute_dataset_state_stats(data_paths)
        logger.info(f"Computed state stats: min={state_stats['state_min']}, max={state_stats['state_max']}")
    else:
        logger.info("Using default state range [-3, 3]")

    # Build episode table once on the main process and pass to workers
    logger.info("Building episode table ...")
    temp_dataset = VivaDataset(
        data_paths=data_paths,
        video_height=config.common.video_height,
        video_width=config.common.video_width,
        state_stats=state_stats,
    )
    episode_table = build_episode_table(temp_dataset)
    del temp_dataset

    # Keep only first N episodes per sub-dataset (grouped by sub_ds_idx)
    if args.max_episodes_per_dataset is not None:
        from collections import defaultdict
        counter = defaultdict(int)
        filtered = []
        for ep in episode_table:
            idx = ep["sub_ds_idx"]
            if counter[idx] < args.max_episodes_per_dataset:
                filtered.append(ep)
                counter[idx] += 1
        logger.info(
            f"  max_episodes_per_dataset={args.max_episodes_per_dataset}: "
            f"{len(episode_table)} -> {len(filtered)} episodes"
        )
        episode_table = filtered

    total_episodes = len(episode_table)
    total_frames = sum(ep["episode_length"] for ep in episode_table)
    logger.info(f"  Total episodes: {total_episodes}")
    logger.info(f"  Total frames:   {total_frames}")

    # Choose launch mode
    launch_mode = args.launch_mode
    if launch_mode == "auto":
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            launch_mode = "torchrun"
        elif args.num_gpus == 1:
            launch_mode = "single"
        else:
            launch_mode = "spawn"

    logger.info(f"  Launch mode: {launch_mode}")
    logger.info("=" * 60)

    if launch_mode == "torchrun":
        run_distributed(args, config, data_paths, episode_table, state_stats)
    elif launch_mode == "spawn":
        num_gpus = min(args.num_gpus, torch.cuda.device_count())
        logger.info(f"Spawning {num_gpus} worker processes ...")
        run_mp_spawn(args, config, data_paths, episode_table, num_gpus, state_stats)
    else:
        run_single_gpu(args, config, data_paths, episode_table, state_stats)
        merge_summaries(args.output_dir, 1)

    logger.info("All done!")


if __name__ == "__main__":
    main()

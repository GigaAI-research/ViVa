"""
visualization.py

Run frame-by-frame inference on a specified episode using VivaModel,
and generate a composite video with the original video on top and
a real-time value curve below.

Usage examples
    python visualization.py --data_path /path/to/dataset --checkpoint /path/to/checkpoint --config /path/to/config.yaml --episode 0

Note: --config is required because it controls whether to compute training-set state statistics , 
which are needed for min-max normalization of the state vectors.
"""

import os
import argparse
import ast
import logging
import numpy as np
import torch
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import av
from omegaconf import OmegaConf

from viva_model import VivaModel
from viva_dataset import VivaDataset, compute_dataset_state_stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ────────────────── Model Loading ──────────────────

def load_model(checkpoint_dir: str, config, device: str = "cuda"):
    """Load VivaModel from an Accelerate checkpoint."""
    from safetensors.torch import load_file

    logger.info(f"Loading model from {checkpoint_dir}")
    model = VivaModel(config)

    model_file = None
    for name in ["model.safetensors", "pytorch_model.bin"]:
        path = os.path.join(checkpoint_dir, name)
        if os.path.exists(path):
            model_file = path
            break

    if model_file is None:
        raise FileNotFoundError(f"No model weights found in {checkpoint_dir}")

    logger.info(f"Loading weights from {model_file}")
    if model_file.endswith(".safetensors"):
        state_dict = load_file(model_file)
    else:
        state_dict = torch.load(model_file, map_location="cpu")

    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    return model


# ────────────────── Data Loading ──────────────────

def build_episode_table(dataset: VivaDataset):
    """Build episode metadata table."""
    table = []
    global_offset = 0
    for sub_ds_idx, sub_lengths in enumerate(dataset._subdataset_episode_lengths):
        for ep_idx, ep_len in enumerate(sub_lengths):
            table.append({
                "sub_ds_idx": sub_ds_idx,
                "episode_index": ep_idx,
                "episode_length": ep_len,
                "global_start": global_offset,
            })
            global_offset += ep_len
    return table


def load_video_frames(video_dir, episode_idx):
    """Load mp4 frames from the observation.images.cam_high directory."""
    path = os.path.join(video_dir, f"episode_{episode_idx:06d}.mp4")
    if not os.path.exists(path):
        logger.warning(f"Video not found: {path}")
        return None
    container = av.open(path)
    return [f.to_rgb().to_ndarray() for f in container.decode(video=0)]


def load_state_txt(path: str):
    """Parse state_stats_3task.txt and return {'state_min': array, 'state_max': array}."""
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


# ────────────────── Single-Episode Inference ──────────────────

@torch.no_grad()
def infer_episode(model, dataset, ep_info, t5_embedding, device, num_inference_steps=1):
    """
    Run frame-by-frame inference for a single episode and return
    an array of predictions (value ∈ [0,1]).
    """
    global_start = ep_info["global_start"]
    ep_len = ep_info["episode_length"]
    dtype = torch.bfloat16

    predictions = []

    for i in range(ep_len):
        global_idx = global_start + i
        sample = dataset.lerobot_dataset[global_idx]

        # state preprocessing (consistent with annotate_v1_fut.py)
        state_raw = sample.get("observation.state", None)
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
        cam_high_raw = sample.get("observation.images.cam_high")
        cam_left_raw = sample.get("observation.images.cam_left_wrist")
        cam_right_raw = sample.get("observation.images.cam_right_wrist")

        cam_high = dataset._process_image(cam_high_raw) if cam_high_raw is not None else torch.zeros(3, dataset.video_height, dataset.video_width)
        cam_left = dataset._process_image(cam_left_raw) if cam_left_raw is not None else torch.zeros(3, dataset.video_height, dataset.video_width)
        cam_right = dataset._process_image(cam_right_raw) if cam_right_raw is not None else torch.zeros(3, dataset.video_height, dataset.video_width)

        # batch dim = 1
        state_b = state.unsqueeze(0).to(device=device, dtype=dtype)
        cam_high_b = cam_high.unsqueeze(0).to(device=device, dtype=dtype)
        cam_left_b = cam_left.unsqueeze(0).to(device=device, dtype=dtype)
        cam_right_b = cam_right.unsqueeze(0).to(device=device, dtype=dtype)

        if t5_embedding is not None:
            t5_emb = t5_embedding.unsqueeze(0).to(device=device, dtype=dtype)
        else:
            t5_emb = torch.zeros(1, 4, 4096, device=device, dtype=dtype)

        output = model.predict_value(
            state_b, cam_left_b, cam_right_b, cam_high_b,
            t5_emb, num_inference_steps=num_inference_steps,
        )
        pred_value = output["value"][0].float().item()
        predictions.append(pred_value)

    return np.array(predictions, dtype=np.float32)


# ────────────────── Visualization Rendering ──────────────────

def render_video(frames, predictions, output_path, fps=30, title=""):
    """
    Display video frames on top and the evolving value curve below.
    """
    T = min(len(frames), len(predictions))
    frames = frames[:T]
    predictions = predictions[:T]
    h_img, w_img = frames[0].shape[:2]

    # Canvas size: video height + curve height (half of video height)
    fig_h = (h_img + h_img // 2) / 100
    fig_w = w_img / 100
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=100)
    fig.subplots_adjust(right=0.82)  # leave room for val=xxx on the right

    gs = fig.add_gridspec(2, 1, hspace=0.25, height_ratios=[2, 1])

    # Top: video
    ax_video = fig.add_subplot(gs[0, 0])
    ax_video.axis("off")
    img_handle = ax_video.imshow(frames[0])
    if title:
        ax_video.set_title(title, fontsize=10)

    # bottom: value curve
    ax_plot = fig.add_subplot(gs[1, 0])
    ax_plot.set_xlim(0, T)
    ax_plot.set_ylim(-1.05, 0.05)
    ax_plot.set_xlabel("Timestep", fontsize=8)
    ax_plot.set_ylabel("Value", fontsize=8)
    ax_plot.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.3)
    ax_plot.axhline(-1, color="black", linewidth=0.5, linestyle="--", alpha=0.3)

    xs = np.arange(T)
    neg_preds = -predictions
    # gray static background: full prediction (negated, 0 at top, -1 at bottom)
    ax_plot.plot(xs, neg_preds, color="lightgray", alpha=0.4, linewidth=0.8)
    # green dynamic line: portion progressed so far
    line_cur, = ax_plot.plot([], [], color="green", linewidth=1.5)
    # vertical dashed line marking the current frame
    vline = ax_plot.axvline(0, linestyle="--", color="gray", linewidth=0.8, alpha=0.6)

    W, H = fig.canvas.get_width_height()
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (W, H),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open VideoWriter for {output_path}")

    for t in range(T):
        img_handle.set_data(frames[t])
        line_cur.set_data(xs[: t + 1], neg_preds[: t + 1])
        vline.set_xdata([t, t])

        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf = buf.reshape(H, W, 3)
        writer.write(cv2.cvtColor(buf, cv2.COLOR_RGB2BGR))

    writer.release()
    plt.close(fig)
    logger.info(f"Video saved to {output_path}")


# ────────────────── Main Function ──────────────────

def main():
    parser = argparse.ArgumentParser(
        description="VivaModel single-episode real-time inference visualization"
    )
    parser.add_argument(
        "--data_path", type=str, required=True,
        help="LeRobot data root directory (must contain data/chunk-000 and videos/chunk-000)"
    )
    parser.add_argument(
        "--episode", type=int, default=0,
        help="Episode index to visualize, default 0"
    )
    parser.add_argument(
        "--checkpoint", type=str,
        default="./checkpoints/3task/checkpoint_step_45069",
        help="Model checkpoint directory"
    )
    parser.add_argument(
        "--config", type=str,
        default="config/train_8gpu_3task.yaml",
        help="config file. It is used here to obtain dataset settings (e.g., compute_state_stats) , so that the training-set state statistics can be computed for min-max normalization."
    )
    parser.add_argument(
        "--t5_embedding", type=str,
        default="/shared_disk/users/hao.li/policy/data/t5_stack_box.pt",
        help="Path to T5 embedding"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output mp4 path, default <script_dir>/result/episode_{episode:06d}.mp4"
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_inference_steps", type=int, default=1)
    parser.add_argument(
        "--state_txt", type=str, default=None,
        help="Path to pre-computed state stats file ; will be read directly if provided"
    )
    args = parser.parse_args()

    # --- Load config and model ---
    config = OmegaConf.load(args.config)
    model = load_model(args.checkpoint, config, args.device)

    # --- Load T5 embedding ---
    t5_embedding = None
    if args.t5_embedding and os.path.exists(args.t5_embedding):
        t5_embedding = torch.load(args.t5_embedding, map_location="cpu")
        logger.info(f"T5 embedding loaded: {t5_embedding.shape}")

    # --- Determine state stats ---
    # Priority: 1) --state_txt  2) auto-compute if compute_state_stats=true in config  3) default [-3, 3]
    state_stats = None
    if args.state_txt and os.path.exists(args.state_txt):
        state_stats = load_state_txt(args.state_txt)
        logger.info(f"Loaded state stats from {args.state_txt}")
        logger.info(f"  min: {state_stats['state_min']}")
        logger.info(f"  max: {state_stats['state_max']}")
    elif config.dataset.get("compute_state_stats", False):
        logger.info("Computing state statistics from data_path (compute_state_stats=true in config)...")
        state_stats = compute_dataset_state_stats([args.data_path])
        logger.info(f"Computed state stats: min={state_stats['state_min']}, max={state_stats['state_max']}")
    else:
        logger.info("Using default state range [-3, 3]")

    # --- Load dataset and locate target episode ---
    dataset = VivaDataset(
        data_paths=[args.data_path],
        video_height=config.common.video_height,
        video_width=config.common.video_width,
        state_stats=state_stats,
    )
    episode_table = build_episode_table(dataset)

    ep_info = None
    for ep in episode_table:
        if ep["episode_index"] == args.episode:
            ep_info = ep
            break
    if ep_info is None:
        available = [ep["episode_index"] for ep in episode_table]
        raise ValueError(f"Episode {args.episode} not found. Available: {available}")

    logger.info(
        f"Processing episode {args.episode}, length={ep_info['episode_length']}"
    )

    # --- Frame-by-frame inference ---
    predictions = infer_episode(
        model, dataset, ep_info, t5_embedding,
        args.device, args.num_inference_steps,
    )
    logger.info(
        f"Inference done. predictions range: "
        f"[{predictions.min():.4f}, {predictions.max():.4f}]"
    )

    # --- Load original video frames ---
    video_dir = os.path.join(
        args.data_path, "videos/chunk-000/observation.images.cam_high"
    )
    frames = load_video_frames(video_dir, args.episode)
    if frames is None:
        raise RuntimeError("Failed to load video frames for visualization")

    # --- Determine output path ---
    if args.output is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, "result")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"episode_{args.episode:06d}.mp4")
    else:
        output_path = args.output
        parent = os.path.dirname(output_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    # --- Render video ---
    title = f"Ep{args.episode}"
    render_video(frames, predictions, output_path, fps=args.fps, title=title)
    logger.info("All done!")


if __name__ == "__main__":
    main()

"""Convert RoboChallenge Table30v2 ALOHA-style episodes to LeRobot format.

The shipped `convert_to_lerobot.py` targets ARX5 single-arm with feature keys
`global_image` / `wrist_image`. This script targets ALOHA dual-arm and writes
features under the names ViVa expects:

  observation.images.cam_high
  observation.images.cam_left_wrist
  observation.images.cam_right_wrist
  observation.state          # 14 = (left.joints[6] + left.gripper[1]) +
                             #      (right.joints[6] + right.gripper[1])
  action                     # 14, same layout (action == proprio for value-only training)

Stored as `dtype: "video"` to keep on-disk size small.

Usage:
    export HF_LEROBOT_HOME=$PWD/data/lerobot_home
    python data/table30v2/convert_to_lerobot_aloha.py \
        --repo-name fold_clothes_aloha \
        --raw-dataset data/table30v2/fold_the_clothes \
        --frame-interval 1
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _state_vec_14(left_row: dict, right_row: dict) -> np.ndarray:
    left_jp = np.asarray(left_row["joint_positions"], dtype=np.float32)
    right_jp = np.asarray(right_row["joint_positions"], dtype=np.float32)
    left_g = np.float32(left_row["gripper_width"])
    right_g = np.float32(right_row["gripper_width"])
    if left_jp.shape != (6,) or right_jp.shape != (6,):
        raise ValueError(
            f"expected 6-dim joint_positions, got "
            f"left={left_jp.shape}, right={right_jp.shape}"
        )
    return np.concatenate([left_jp, [left_g], right_jp, [right_g]]).astype(np.float32)


def create_dataset(repo_name: str, robot_type: str, fps: float, h: int, w: int) -> LeRobotDataset:
    return LeRobotDataset.create(
        repo_id=repo_name,
        robot_type=robot_type,
        fps=int(round(fps)),
        features={
            "observation.images.cam_high": {
                "dtype": "video",
                "shape": (h, w, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.images.cam_left_wrist": {
                "dtype": "video",
                "shape": (h, w, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.images.cam_right_wrist": {
                "dtype": "video",
                "shape": (h, w, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (14,),
                "names": ["state"],
            },
            "action": {
                "dtype": "float32",
                "shape": (14,),
                "names": ["action"],
            },
        },
        image_writer_threads=8,
        image_writer_processes=4,
    )


def process_episode(
    episode_path: Path,
    dataset: LeRobotDataset,
    frame_interval: int,
    prompt: str,
) -> None:
    states_dir = episode_path / "states"
    videos_dir = episode_path / "videos"

    left_states = _load_jsonl(states_dir / "left_states.jsonl")
    right_states = _load_jsonl(states_dir / "right_states.jsonl")
    if len(left_states) != len(right_states):
        raise ValueError(
            f"{episode_path.name}: left/right state length mismatch "
            f"({len(left_states)} vs {len(right_states)})"
        )

    cam_high = cv2.VideoCapture(str(videos_dir / "cam_high_rgb.mp4"))
    cam_left = cv2.VideoCapture(str(videos_dir / "cam_left_wrist_rgb.mp4"))
    cam_right = cv2.VideoCapture(str(videos_dir / "cam_right_wrist_rgb.mp4"))

    n_frames = min(
        int(cam_high.get(cv2.CAP_PROP_FRAME_COUNT)),
        int(cam_left.get(cv2.CAP_PROP_FRAME_COUNT)),
        int(cam_right.get(cv2.CAP_PROP_FRAME_COUNT)),
        len(left_states),
    )
    if n_frames < 2:
        cam_high.release(); cam_left.release(); cam_right.release()
        return

    written = 0
    for idx in range(0, n_frames, frame_interval):
        ok_h, fr_h = cam_high.read()
        ok_l, fr_l = cam_left.read()
        ok_r, fr_r = cam_right.read()
        if not (ok_h and ok_l and ok_r):
            break
        fr_h = cv2.cvtColor(fr_h, cv2.COLOR_BGR2RGB)
        fr_l = cv2.cvtColor(fr_l, cv2.COLOR_BGR2RGB)
        fr_r = cv2.cvtColor(fr_r, cv2.COLOR_BGR2RGB)

        s = _state_vec_14(left_states[idx], right_states[idx])

        dataset.add_frame(
            {
                "observation.images.cam_high": fr_h,
                "observation.images.cam_left_wrist": fr_l,
                "observation.images.cam_right_wrist": fr_r,
                "observation.state": s,
                "action": s,
            },
            task=prompt,
        )
        written += 1

        if frame_interval > 1:
            for _ in range(frame_interval - 1):
                cam_high.read(); cam_left.read(); cam_right.read()

    cam_high.release(); cam_left.release(); cam_right.release()
    if written > 0:
        dataset.save_episode()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-name", required=True)
    ap.add_argument("--raw-dataset", required=True, type=str)
    ap.add_argument("--frame-interval", type=int, default=1)
    ap.add_argument("--overwrite-repo", action="store_true")
    ap.add_argument("--max-episodes", type=int, default=None,
                    help="Cap number of episodes converted (smoke runs).")
    args = ap.parse_args()

    raw = Path(args.raw_dataset)
    dst = HF_LEROBOT_HOME / args.repo_name
    if args.overwrite_repo and dst.exists():
        print(f"removing existing dataset at {dst}")
        shutil.rmtree(dst)

    info = json.loads((raw / "meta" / "task_info.json").read_text(encoding="utf-8"))
    prompt = info["task_desc"]["prompt"]
    fps = float(info["video_info"]["fps"])
    task_tags = info["task_desc"].get("task_tag", [])
    robot_type = next(
        (t for t in task_tags if t in {"ARX5", "UR5", "ALOHA", "DOS-W1"}),
        "ALOHA",
    )

    sample_ep = next((p for p in (raw / "data").iterdir() if p.is_dir()), None)
    if sample_ep is None:
        raise SystemExit(f"No episodes under {raw / 'data'}")
    probe = cv2.VideoCapture(str(sample_ep / "videos" / "cam_high_rgb.mp4"))
    h = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH))
    probe.release()
    if h <= 0 or w <= 0:
        raise SystemExit("could not read cam_high_rgb.mp4 dimensions")
    print(f"robot: {robot_type} | video {h}x{w} @ {fps}fps | prompt: {prompt}")

    dataset = create_dataset(args.repo_name, robot_type=robot_type, fps=fps, h=h, w=w)
    episodes = sorted(p for p in (raw / "data").iterdir() if p.is_dir())
    if args.max_episodes is not None:
        episodes = episodes[: args.max_episodes]
    for i, ep in enumerate(episodes):
        print(f"  [{i + 1}/{len(episodes)}] {ep.name}")
        process_episode(ep, dataset, args.frame_interval, prompt)

    print(f"done. dataset at: {dst}")


if __name__ == "__main__":
    main()

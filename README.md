# ViVa
> **ViVa: A Video-Generative Value Model for Robot Reinforcement Learning**

[![arXiv](https://img.shields.io/badge/arXiv-2604.08168-b31b1b.svg)](http://arxiv.org/abs/2604.08168)
[![Website](https://img.shields.io/badge/Website-Project_Page-blue.svg)](https://viva-value-model.github.io/)

## Overview
ViVa is a video-generative value model that repurposes pretrained video generation backbones for value estimation in robot reinforcement learning. Instead of relying on static visual understanding, it models value as a function of anticipated future dynamics by jointly predicting future proprioceptive states and a scalar value from current observations. By leveraging spatiotemporal priors learned from large-scale video data, ViVa grounds value estimation in predicted embodiment evolution, enabling more reliable progress assessment in long-horizon tasks. Integrated into RL pipelines (e.g., RECAP), it provides improved policy optimization, stronger sensitivity to execution errors, and better generalization to novel objects compared to VLM-based value models.

## Training
### Environment Dependencies
- [ ] Environment setup guide is in progress (TODO).

### WAN Pretrained Weights
Download the pretrained WAN `2.2-TI2V-5B` weights from Hugging Face and place them under `weights/Wan2.2-TI2V-5B/`:

```bash
huggingface-cli download Wan-AI/Wan2.2-TI2V-5B --local-dir weights/Wan2.2-TI2V-5B --local-dir-use-symlinks False
```

### T5 Embedding
After configuring the LeRobot dataset path and model weights path, generate offline T5 embeddings for task descriptions:

```bash
python get_text_embedding.py
```

### Start Training (3 Tasks)
After setting the corresponding LeRobot dataset path and T5 embedding path, start training:

```bash
bash train_3task.sh
```

- Log: `logs/train_3task.log`
- Checkpoint: `checkpoints/3task/`
- Tmux session: `train_3task`

### Monitor Training

```bash
tail -f logs/train_3task.log
tmux attach -t train_3task
```

---

### Citation

```bibtex
@article{viva2026,
  title={ViVa: A Video-Generative Value Model for Robot Reinforcement Learning},
  author={Lv, Jindi and Li, Hao and Li, Jie and Nie, Yifei and Kong, Fankun and Wang, Yang and Wang, Xiaofeng and Zhu, Zheng and Ni, Chaojun and Deng, Qiuping and Li, Hengtao and Lv, Jiancheng and Huang, Guan},
  year={2026},
  url={http://arxiv.org/abs/2604.08168}
}
```

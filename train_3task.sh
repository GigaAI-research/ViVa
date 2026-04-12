# Training script for VivaModel - 3 Tasks
# 7-frame: [blank, state, cam_left, cam_right, cam_high, future_state, value]
# future_offset: 50 frames
         

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$SCRIPT_DIR"

FUTURE_STATE_WEIGHT="0.5"
VALUE_WEIGHT="1.0"

CONFIG_PATH="./config/train_8gpu_3task.yaml"
LOG_DIR="./logs"
CHECKPOINT_DIR="./checkpoints/3task"
LOG_FILE="${LOG_DIR}/train_3task.log"
TMUX_SESSION="train_3task"

echo "=========================================="
echo "VivaModel Training - 3 Tasks"
echo "=========================================="
echo "Config:                  $CONFIG_PATH"
echo "Checkpoint base:         $CHECKPOINT_DIR"
echo "loss_weight_future_state: $FUTURE_STATE_WEIGHT"
echo "loss_weight_value:       $VALUE_WEIGHT (from yaml)"
echo "Run name suffix:         3task"
echo "Log:                     $LOG_FILE"
echo "GPUs:                    0,1,2,3,4,5,6,7"
echo "Tmux session:            $TMUX_SESSION"
echo "=========================================="

mkdir -p "$LOG_DIR"
mkdir -p "$CHECKPOINT_DIR"


tmux kill-session -t "$TMUX_SESSION" 2>/dev/null || true

tmux new-session -d -s "$TMUX_SESSION" bash -c "
cd "$SCRIPT_DIR" && \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
    --num_processes 8 \
    --mixed_precision bf16 \
    train.py \
    --config $CONFIG_PATH \
    --checkpoint_dir $CHECKPOINT_DIR \
    --loss_weight_future_state $FUTURE_STATE_WEIGHT \
    2>&1 | tee $LOG_FILE
"

echo ""
echo "Training started in tmux session: $TMUX_SESSION"
echo "Attach:  tmux attach -t $TMUX_SESSION"
echo "Monitor: tail -f $LOG_FILE"

#!/bin/bash
# Example script demonstrating how to stop training, save the model and replay buffer,
# and then resume training.

# Set environment variables
ENV="hopper-hop"
SEED=42
SAVE_PATH="./saved_models"
TOTAL_TIMESTEPS=1000000
SAVE_INTERVAL=100000

# Step 1: Start training and stop after SAVE_INTERVAL timesteps
echo "Starting training and stopping after $SAVE_INTERVAL timesteps..."
python train_dmc.py \
    --env $ENV \
    --seed $SEED \
    --total_timesteps $SAVE_INTERVAL \
    --save_path $SAVE_PATH \
    --stop_and_save

# Step 2: Resume training from the saved model and replay buffer
echo "Resuming training from saved model and replay buffer..."
python train_dmc.py \
    --env $ENV \
    --seed $SEED \
    --total_timesteps $TOTAL_TIMESTEPS \
    --resume \
    --model_path "$SAVE_PATH/SAC_model_${SAVE_INTERVAL}.zip"

echo "Training completed!" 
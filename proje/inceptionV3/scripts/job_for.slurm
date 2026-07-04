#!/bin/bash

#SBATCH --job-name=lightning-inceptionv3 # Changed job name
#SBATCH --output=./out/%x.%j.out
#SBATCH --error=./out/%x.%j.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=20 # Ensure this matches num_workers and data loading needs
#SBATCH --time=0-04:00:00 # Adjusted time, InceptionV3 is larger. This is for ALL runs.
#SBATCH --partition=barbun-cuda
#SBATCH --account=egitim
#SBATCH --reservation=yzup-ders

mkdir -p ./out

module load apps/truba-ai/gpu-2024.0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# For PyTorch DDP, NCCL environment variables might be needed on some systems,
# but PyTorch Lightning usually handles this.
# export NCCL_DEBUG=INFO # Optional: for debugging NCCL
# export NCCL_SOCKET_IFNAME=ib0 # Or your specific InfiniBand interface

# Parameters
EPOCHS=10 # Or more for InceptionV3 to converge well
GPUS=$SLURM_GPUS_PER_NODE
NODES=$SLURM_NNODES
BASE_LR=1e-4 # Base learning rate

declare -a strategies=("ddp" "fsdp") # Reduced for example, add "fsdp1", "fsdp2" as needed
declare -a optimizers=("adamw" "sgd")   # Reduced for example, add "adam"
declare -a batch_sizes=(32 64)       # Adjusted batch sizes, 128 might be too large for InceptionV3 on some GPUs

# LR Scheduler yapılandırmaları ("reduce" eklendi)
declare -a lr_schedulers=("none" "step" "cosine" "reduce")
# StepLR için
LR_STEP_SIZE=5  # For StepLR: Number of epochs between LR decay
LR_GAMMA=0.1    # For StepLR: Multiplicative factor of LR decay
# ReduceLROnPlateau için
LR_REDUCE_FACTOR=0.1
LR_REDUCE_PATIENCE=5 # Daha agresif bir deneme için 5, normalde 10 olabilir
LR_MONITOR_METRIC="val_loss" # İzlenecek metrik (val_loss veya val_acc)

echo "Running InceptionV3 experiments with ReduceLROnPlateau option"
echo "=================================================================="

for strategy in "${strategies[@]}"; do
  for optimizer in "${optimizers[@]}"; do
    for batch_size in "${batch_sizes[@]}"; do
      for lr_scheduler in "${lr_schedulers[@]}"; do
        echo "---------------------------------------------------------------------------------"
        echo "Running InceptionV3 with strategy=$strategy, optimizer=$optimizer, batch_size=$batch_size, lr_scheduler=$lr_scheduler, epochs=$EPOCHS, base_lr=$BASE_LR"
        echo "---------------------------------------------------------------------------------"
        
        # Construct the command
        CMD="srun python3 cifar10_lightning.py \
          --gpus=$GPUS \
          --nodes=$NODES \
          --epochs=$EPOCHS \
          --batch_size=$batch_size \
          --accelerator=gpu \
          --strategy=$strategy \
          --optimizer=$optimizer \
          --learning_rate=$BASE_LR \
          --lr_scheduler_name=$lr_scheduler"

        if [ "$lr_scheduler" == "step" ]; then
          CMD="$CMD --lr_step_size=$LR_STEP_SIZE --lr_gamma=$LR_GAMMA"
        elif [ "$lr_scheduler" == "cosine" ]; then
          # Python script defaults T_max to EPOCHS if --lr_t_max is not provided or is -1.
          # To explicitly set T_max from here, uncomment and set:
          # CMD="$CMD --lr_t_max=$EPOCHS"
          : # No specific params to add here if using default T_max logic in script
        elif [ "$lr_scheduler" == "reduce" ]; then
          CMD="$CMD --lr_reduce_factor=$LR_REDUCE_FACTOR \
                     --lr_reduce_patience=$LR_REDUCE_PATIENCE \
                     --lr_monitor_metric=$LR_MONITOR_METRIC"
        fi
        
        echo "Executing: $CMD"
        eval $CMD
        echo "Finished run for InceptionV3: strategy=$strategy, optimizer=$optimizer, batch_size=$batch_size, lr_scheduler=$lr_scheduler"
        echo "---------------------------------------------------------------------------------"
        sleep 10 # Small delay between runs, if necessary
      done
    done
  done
done

echo "All InceptionV3 hyperparameter tuning runs completed."
exit
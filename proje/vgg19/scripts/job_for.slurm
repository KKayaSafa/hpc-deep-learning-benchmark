#!/bin/bash

#SBATCH --job-name=lightning-vgg19 # Changed job name
#SBATCH --output=./out/%x.%j.out
#SBATCH --error=./out/%x.%j.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=20 # Ensure this matches num_workers and data loading needs
#SBATCH --time=0-04:00:00 # VGG19 is smaller than InceptionV3, should be faster
#SBATCH --partition=barbun-cuda
#SBATCH --account=egitim
#SBATCH --reservation=yzup-ders

mkdir -p ./out

module load apps/truba-ai/gpu-2024.0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Parameters
EPOCHS=10 # VGG19 converges faster than InceptionV3
GPUS=$SLURM_GPUS_PER_NODE
NODES=$SLURM_NNODES
BASE_LR=1e-4 # Base learning rate

declare -a strategies=("ddp" "fsdp") # VGG19 is smaller, both strategies should work well
declare -a optimizers=("adamw" "sgd")   # Both optimizers work well with VGG
declare -a batch_sizes=(32 64 128)    # VGG19 allows larger batch sizes than InceptionV3

# LR Scheduler yapılandırmaları ("reduce" eklendi)
declare -a lr_schedulers=("none" "step" "cosine" "reduce")
# StepLR için
LR_STEP_SIZE=5  # For StepLR: Number of epochs between LR decay
LR_GAMMA=0.1    # For StepLR: Multiplicative factor of LR decay
# ReduceLROnPlateau için
LR_REDUCE_FACTOR=0.1
LR_REDUCE_PATIENCE=5 # Daha agresif bir deneme için 5, normalde 10 olabilir
LR_MONITOR_METRIC="val_loss" # İzlenecek metrik (val_loss veya val_acc)

echo "Running VGG19 experiments with ReduceLROnPlateau option"
echo "=================================================================="

for strategy in "${strategies[@]}"; do
  for optimizer in "${optimizers[@]}"; do
    for batch_size in "${batch_sizes[@]}"; do
      for lr_scheduler in "${lr_schedulers[@]}"; do
        echo "---------------------------------------------------------------------------------"
        echo "Running VGG19 with strategy=$strategy, optimizer=$optimizer, batch_size=$batch_size, lr_scheduler=$lr_scheduler, epochs=$EPOCHS, base_lr=$BASE_LR"
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
        echo "Finished run for VGG19: strategy=$strategy, optimizer=$optimizer, batch_size=$batch_size, lr_scheduler=$lr_scheduler"
        echo "---------------------------------------------------------------------------------"
        sleep 10 # Small delay between runs, if necessary
      done
    done
  done
done

echo "All VGG19 hyperparameter tuning runs completed."
exit
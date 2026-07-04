#!/bin/bash

#SBATCH --job-name=2GPU-efficientnetb2-barbun # Changed job name
#SBATCH --output=./out/%x.%j.out
#SBATCH --error=./out/%x.%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=20 # Ensure this matches num_workers and data loading needs
#SBATCH --time=0-04:00:00 # EfficientNetB2 is efficient, should be faster than InceptionV3
#SBATCH --partition=barbun-cuda
#SBATCH --account=egitim
#SBATCH --reservation=yzup-ders

mkdir -p ./out

module load apps/truba-ai/gpu-2024.0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Parameters
EPOCHS=10 # EfficientNet benefits from more epochs due to compound scaling
GPUS=$SLURM_GPUS_PER_NODE
NODES=$SLURM_NNODES
BASE_LR=1e-3 # EfficientNet often benefits from slightly higher learning rates

declare -a strategies=("fsdp1" "fsdp2" "fsdp") # Both strategies work well with EfficientNet
declare -a optimizers=("rmsprop" "adam" "sgd" "adamw")   # RMSprop eklendi
declare -a batch_sizes=(64 128)       # Updated batch sizes as requested

# LR Scheduler yapılandırmaları ("reduce" eklendi)
declare -a lr_schedulers=("none" "step" "cosine" "reduce")
# StepLR için
LR_STEP_SIZE=5  # For StepLR: Number of epochs between LR decay
LR_GAMMA=0.1    # For StepLR: Multiplicative factor of LR decay
# ReduceLROnPlateau için
LR_REDUCE_FACTOR=0.1
LR_REDUCE_PATIENCE=3 # EfficientNet converges faster, so shorter patience
LR_MONITOR_METRIC="val_loss" # İzlenecek metrik (val_loss veya val_acc)

# RMSprop parametreleri - EfficientNetB2 için optimize edilmiş
RMSPROP_ALPHA=0.9      # EfficientNet için biraz daha düşük alpha
RMSPROP_EPS=1e-8
RMSPROP_WEIGHT_DECAY=1e-5  # Çok hafif weight decay (EfficientNet zaten efficient)
RMSPROP_MOMENTUM=0.9       # Momentum ekleyerek daha kararlı eğitim

echo "Running EfficientNetB2 experiments with RMSprop option"
echo "=================================================================="

for strategy in "${strategies[@]}"; do
  for optimizer in "${optimizers[@]}"; do
    for batch_size in "${batch_sizes[@]}"; do
      for lr_scheduler in "${lr_schedulers[@]}"; do
        echo "---------------------------------------------------------------------------------"
        echo "Running EfficientNetB2 with strategy=$strategy, optimizer=$optimizer, batch_size=$batch_size, lr_scheduler=$lr_scheduler, epochs=$EPOCHS, base_lr=$BASE_LR"
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

        # RMSprop parametrelerini ekle
        if [ "$optimizer" == "rmsprop" ]; then
          CMD="$CMD --rmsprop_alpha=$RMSPROP_ALPHA \
                     --rmsprop_eps=$RMSPROP_EPS \
                     --rmsprop_weight_decay=$RMSPROP_WEIGHT_DECAY \
                     --rmsprop_momentum=$RMSPROP_MOMENTUM"
        fi

        # LR Scheduler parametrelerini ekle
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
        echo "Finished run for EfficientNetB2: strategy=$strategy, optimizer=$optimizer, batch_size=$batch_size, lr_scheduler=$lr_scheduler"
        echo "---------------------------------------------------------------------------------"
        sleep 10 # Small delay between runs, if necessary
      done
    done
  done
done

echo "All EfficientNetB2 hyperparameter tuning runs completed."
exit

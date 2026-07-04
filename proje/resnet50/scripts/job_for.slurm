#!/bin/bash

#SBATCH --job-name=lightning-resnet50
#SBATCH --output=./out/%x.%j.out
#SBATCH --error=./out/%x.%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=10
#SBATCH --time=0-02:00:00
#SBATCH --partition=akya-cuda
#SBATCH --account=egitim
#SBATCH --reservation=yzup-ders

mkdir -p ./out

module load apps/truba-ai/gpu-2024.0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

EPOCHS=10
GPUS=$SLURM_GPUS_PER_NODE
NODES=$SLURM_NNODES
BASE_LR=1e-4

declare -a strategies=("ddp" "fsdp" "fsdp1" "fsdp2")
declare -a optimizers=("sgd" "adam" "adamw")
declare -a batch_sizes=(32 128)

# LR Scheduler yapılandırmaları ("reduce" eklendi)
declare -a lr_schedulers=("none" "step" "cosine" "reduce")
# StepLR için
LR_STEP_SIZE=5
LR_GAMMA=0.1
# ReduceLROnPlateau için
LR_REDUCE_FACTOR=0.1
LR_REDUCE_PATIENCE=5 # Daha agresif bir deneme için 5, normalde 10 olabilir
LR_MONITOR_METRIC="val_loss" # İzlenecek metrik (val_loss veya val_acc)


echo "Running ResNet50 experiments with ReduceLROnPlateau option"
echo "=================================================================="

for strategy in "${strategies[@]}"; do
  for optimizer in "${optimizers[@]}"; do
    for batch_size in "${batch_sizes[@]}"; do
      for lr_scheduler in "${lr_schedulers[@]}"; do
        echo "---------------------------------------------------------------------------------"
        echo "Running ResNet50 with strategy=$strategy, optimizer=$optimizer, batch_size=$batch_size, lr_scheduler=$lr_scheduler, epochs=$EPOCHS, base_lr=$BASE_LR"
        echo "---------------------------------------------------------------------------------"
        
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
          # T_max Python betiğinde varsayılan olarak EPOCHS'a ayarlanır
           CMD="$CMD --lr_t_max=$EPOCHS" # İsterseniz buradan da belirleyebilirsiniz
        elif [ "$lr_scheduler" == "reduce" ]; then
          CMD="$CMD --lr_reduce_factor=$LR_REDUCE_FACTOR \
                     --lr_reduce_patience=$LR_REDUCE_PATIENCE \
                     --lr_monitor_metric=$LR_MONITOR_METRIC"
        fi
        
        echo "Executing: $CMD"
        eval $CMD
        echo "Finished run for ResNet50: strategy=$strategy, optimizer=$optimizer, batch_size=$batch_size, lr_scheduler=$lr_scheduler"
        echo "---------------------------------------------------------------------------------"
        sleep 5 # Çalıştırmalar arasında kısa bir bekleme
      done
    done
  done
done

echo "All ResNet50 hyperparameter tuning runs completed."
exit
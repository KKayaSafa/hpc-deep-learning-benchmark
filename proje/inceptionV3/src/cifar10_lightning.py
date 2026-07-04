import argparse
import os
import shutil
import time
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchvision.models import inception_v3 # Changed from resnet50
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as L
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.strategies import FSDPStrategy, DDPStrategy
from pytorch_lightning.callbacks import DeviceStatsMonitor
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.utils._pytree")
warnings.filterwarnings("ignore", category=FutureWarning, module="pytorch_lightning.strategies.fsdp")
warnings.filterwarnings("ignore", category=FutureWarning, module="torchvision.models.inception")  # InceptionV3 warning'ini ignore et

os.makedirs("./out", exist_ok=True)

class LitInceptionV3(L.LightningModule): # Renamed class
    def __init__(self, learning_rate=1e-2, num_classes=10, optimizer_name="sgd",
                 lr_scheduler_name="none", lr_step_size=10, lr_gamma=0.1, lr_t_max=100,
                 lr_reduce_factor=0.1, lr_reduce_patience=10, lr_monitor_metric="val_loss"): # Added ReduceLROnPlateau params
        super().__init__()
        self.save_hyperparameters()

        # Load InceptionV3
        weights_path = './model-weights/inceptionv3_weights.pth' # Updated weights path
        model = inception_v3(weights=None, aux_logits=False, init_weights=True) # Auxiliary logits'i tamamen kapat, init_weights warning'ini kapat
        
        if os.path.exists(weights_path):
            # Pretrained ağırlıkları yükle, ancak aux_logits farklı olduğu için manuel olarak handle et
            try:
                pretrained_state = torch.load(weights_path)
                # Auxiliary classifier parametrelerini kaldır
                pretrained_state = {k: v for k, v in pretrained_state.items() if not k.startswith('AuxLogits')}
                model.load_state_dict(pretrained_state, strict=False)  # strict=False çünkü aux_logits yok
                print(f"Loaded InceptionV3 weights from {weights_path} (without AuxLogits)")
            except Exception as e:
                print(f"Warning: Could not load weights from {weights_path}: {e}")
                print("Using randomly initialized model.")
        else:
            print(f"Warning: InceptionV3 weights not found at {weights_path}. Using randomly initialized model.")
        
        # Modify the main classifier
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
        
        self.model = model

    def forward(self, x):
        # If model.aux_logits was True and model was in training mode,
        # inception_v3 would return InceptionOutputs(logits, aux_logits).
        # Since we set self.model.aux_logits = False, it will always return only the main output.
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images) # This will be a single tensor due to model.aux_logits = False
        loss = F.cross_entropy(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == labels).float().mean()

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        opt_name = self.hparams.optimizer_name.lower()
        if opt_name == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate, momentum=0.9)
        elif opt_name == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        elif opt_name == "adamw":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {opt_name}")

        if self.hparams.lr_scheduler_name == "none":
            return optimizer
        
        scheduler_config = {"interval": "epoch"}
        
        if self.hparams.lr_scheduler_name == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.hparams.lr_step_size, gamma=self.hparams.lr_gamma)
        elif self.hparams.lr_scheduler_name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.lr_t_max)
        elif self.hparams.lr_scheduler_name == "reduce":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min' if 'loss' in self.hparams.lr_monitor_metric else 'max', # Monitör edilen metriğe göre mod ayarı
                factor=self.hparams.lr_reduce_factor,
                patience=self.hparams.lr_reduce_patience,
                verbose=True
            )
            scheduler_config["monitor"] = self.hparams.lr_monitor_metric # ReduceLROnPlateau için monitör metriği
        else:
            raise ValueError(f"Unsupported LR scheduler: {self.hparams.lr_scheduler_name}")
            
        scheduler_config["scheduler"] = scheduler
        return {"optimizer": optimizer, "lr_scheduler": scheduler_config}

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == labels).float().mean()
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True) # val_loss loglanıyor
        self.log("val_acc", acc, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == labels).float().mean()
        self.log("test_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True) # Added prog_bar and sync_dist for consistency
        self.log("test_acc", acc, on_epoch=True, prog_bar=True, sync_dist=True)  # Added prog_bar and sync_dist for consistency
        return loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', default=2, type=int, metavar='N',
                        help='number of GPUs per node')
    parser.add_argument('--nodes', default=1, type=int, metavar='N',
                        help='number of nodes')
    parser.add_argument('--epochs', default=2, type=int, metavar='N',
                        help='maximum number of epochs to run')
    parser.add_argument('--batch_size', default=32, type=int, metavar='N',
                        help='the batch size')
    parser.add_argument('--accelerator', default='gpu', type=str,
                        help='accelerator to use')
    parser.add_argument('--strategy', default='ddp', type=str,
                        help='distributed strategy to use')
    parser.add_argument('--learning_rate', default=1e-4, type=float,
                        help='learning rate')
    parser.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adam", "adamw"],
                    help="Optimizer to use: sgd | adam | adamw")
    
    # LR Scheduler için Argümanlar güncellendi ("reduce" eklendi)
    parser.add_argument('--lr_scheduler_name', type=str, default="none", choices=["none", "step", "cosine", "reduce"], 
                        help='LR Scheduler to use: none | step | cosine | reduce')
    # StepLR için parametreler
    parser.add_argument('--lr_step_size', type=int, default=10, 
                        help='Step size for StepLR scheduler')
    parser.add_argument('--lr_gamma', type=float, default=0.1, 
                        help='Gamma for StepLR scheduler')
    # CosineAnnealingLR için parametreler
    parser.add_argument('--lr_t_max', type=int, default=-1,  # Default to -1, will be set to epochs if not provided for cosine
                        help='T_max for CosineAnnealingLR scheduler (typically number of epochs, -1 for auto epochs)')
    # ReduceLROnPlateau için parametreler
    parser.add_argument('--lr_reduce_factor', type=float, default=0.1, 
                        help='Factor by which the learning rate will be reduced (for ReduceLROnPlateau)')
    parser.add_argument('--lr_reduce_patience', type=int, default=10, 
                        help='Number of epochs with no improvement after which learning rate will be reduced (for ReduceLROnPlateau)')
    parser.add_argument('--lr_monitor_metric', type=str, default="val_loss", 
                        help='Metric to monitor for ReduceLROnPlateau (e.g., val_loss, val_acc)')

    args = parser.parse_args()

    print("Using PyTorch {} and Lightning {}".format(torch.__version__, L.__version__))

    transform = transforms.Compose([
        transforms.Resize(299),  # InceptionV3 expects 299x299 images
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], # Standard ImageNet normalization
                                std=[0.229, 0.224, 0.225])
    ])

    # Veri dizininin varlığını kontrol et
    data_dir = './data'
    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} does not exist. Creating it...")
        os.makedirs(data_dir, exist_ok=True)
    
    print(f"Checking for CIFAR10 dataset in {data_dir}...")
    cifar_dir = os.path.join(data_dir, 'cifar-10-batches-py')
    if not os.path.exists(cifar_dir):
        print(f"CIFAR10 dataset not found in {cifar_dir}. Will download automatically.")
        download_data = True
    else:
        print(f"CIFAR10 dataset found in {cifar_dir}")
        download_data = False

    # Download parametresini dinamik olarak ayarla
    full_train_dataset = CIFAR10(data_dir, train=True, download=download_data, transform=transform)
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
    test_dataset = CIFAR10(data_dir, train=False, download=download_data, transform=transform)

    # Adjust num_workers based on your system's capabilities
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True, shuffle=True, persistent_workers=True if args.accelerator == 'gpu' else False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=2, pin_memory=True, persistent_workers=True if args.accelerator == 'gpu' else False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=2, pin_memory=True, persistent_workers=True if args.accelerator == 'gpu' else False)
    
    # Determine T_max for CosineAnnealingLR if not explicitly set
    actual_lr_t_max = args.lr_t_max
    if args.lr_scheduler_name == "cosine" and args.lr_t_max == -1:
        actual_lr_t_max = args.epochs

    model = LitInceptionV3(learning_rate=args.learning_rate,
                           optimizer_name=args.optimizer,
                           lr_scheduler_name=args.lr_scheduler_name,
                           lr_step_size=args.lr_step_size,
                           lr_gamma=args.lr_gamma,
                           lr_t_max=actual_lr_t_max,
                           lr_reduce_factor=args.lr_reduce_factor,         # Yeni parametreler modele iletiliyor
                           lr_reduce_patience=args.lr_reduce_patience,     # Yeni parametreler modele iletiliyor
                           lr_monitor_metric=args.lr_monitor_metric)       # Yeni parametreler modele iletiliyor
    
    selected_strategy = args.strategy
    if args.strategy == "ddp":
        # DDP için unused parameters'ı handle etmek için
        selected_strategy = DDPStrategy(find_unused_parameters=True)
    elif args.strategy == "fsdp":
        selected_strategy = FSDPStrategy(
            sharding_strategy="FULL_SHARD", # FULL_SHARD, SHARD_GRAD_OP, NO_SHARD
            cpu_offload=False # Consider True if GPU memory is an issue
        )
    elif args.strategy == "fsdp1": # Example: Shard optimizer state and gradients
        selected_strategy = FSDPStrategy(
            sharding_strategy="SHARD_GRAD_OP",
            cpu_offload=False
        )
    elif args.strategy == "fsdp2": # Example: No sharding (similar to DDP)
        selected_strategy = FSDPStrategy(
            sharding_strategy="NO_SHARD",
            cpu_offload=False
        )
    # 'ddp' string will be handled by Trainer directly

    logger = TensorBoardLogger(save_dir="./lightning_logs/", name="experiments_inceptionv3", default_hp_metric=False)
    device_stats = DeviceStatsMonitor()

    checkpoint_dir = os.path.join("./out", f"checkpoints_inceptionv3_{args.strategy}_{args.optimizer}_bs{args.batch_size}_lr{args.learning_rate}_sch{args.lr_scheduler_name}")
    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc", # ReduceLROnPlateau val_loss'u izlese bile, checkpoint val_acc'yi izleyebilir.
        mode="max",
        save_top_k=1,
        save_last=True,
        dirpath=checkpoint_dir,
        filename="inceptionv3-{epoch:02d}-{val_acc:.2f}"
    )

    early_stop_callback = EarlyStopping(
        monitor="val_acc",
        mode="max",
        patience=10, # Increased patience
        verbose=True
    )

    trainer = L.Trainer(
        devices=args.gpus,
        num_nodes=args.nodes,
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        strategy=selected_strategy,
        logger=logger,
        callbacks=[device_stats, early_stop_callback, checkpoint_callback],
        log_every_n_steps=20 # Log every 20 steps
    )

    from datetime import datetime
    t0 = datetime.now()
    trainer.fit(model, train_loader, val_loader)
    dt = datetime.now() - t0
    print(f'Training took {dt}')

    # Load best model for testing
    if checkpoint_callback.best_model_path and os.path.exists(checkpoint_callback.best_model_path):
        print(f"Loading best model from: {checkpoint_callback.best_model_path}")
        trained_model = LitInceptionV3.load_from_checkpoint(checkpoint_callback.best_model_path)
    else:
        print(f"Warning: Best model checkpoint not found at '{checkpoint_callback.best_model_path if checkpoint_callback.best_model_path else 'N/A'}' or path does not exist. Using last trained model for testing.")
        trained_model = model

    test_trainer = L.Trainer(
        devices=1, # Test on a single device
        accelerator=args.accelerator,
        logger=logger
    )

    print("Running test evaluation...")
    test_results = test_trainer.test(trained_model, test_loader)
    
    if not test_results or not isinstance(test_results, list) or not test_results[0]:
        print("Test results are empty or in unexpected format.")
        test_loss, test_acc = -1.0, -1.0
    else:
        test_loss = test_results[0].get("test_loss", -1.0)
        test_acc = test_results[0].get("test_acc", -1.0)
        
    print(f"Test results: Loss={test_loss:.4f}, Acc={test_acc:.4f}")

    hparams_to_log = {
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "model_type": "InceptionV3", # Updated model type
        "gpus": args.gpus,
        "nodes": args.nodes,
        "strategy": args.strategy, # Log the input strategy string
        "optimizer": args.optimizer,
        "lr_scheduler_name": args.lr_scheduler_name
    }
    if args.lr_scheduler_name == "step":
        hparams_to_log["lr_step_size"] = args.lr_step_size
        hparams_to_log["lr_gamma"] = args.lr_gamma
    elif args.lr_scheduler_name == "cosine":
        hparams_to_log["lr_t_max"] = model.hparams.lr_t_max # Get actual T_max from model hparams
    elif args.lr_scheduler_name == "reduce": # ReduceLROnPlateau için hiperparametreler loglanıyor
        hparams_to_log["lr_reduce_factor"] = args.lr_reduce_factor
        hparams_to_log["lr_reduce_patience"] = args.lr_reduce_patience
        hparams_to_log["lr_monitor_metric"] = args.lr_monitor_metric

    logger.log_hyperparams(
        hparams_to_log,
        metrics={
            "test_loss": test_loss,
            "test_acc": test_acc
        }
    )
    logger.save() # Ensure all logs are written

if __name__ == '__main__':
    main()
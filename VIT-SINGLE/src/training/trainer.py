from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import json
from datetime import datetime

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, ReduceLROnPlateau
import torch.optim as optim
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from tqdm import tqdm

from src.config import Config
from src.data.dataset import DeepfakeDataset
from src.models.branch_vit_rgb import RGBViTBranch

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="[%(asctime)s] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)

LINE = "=" * 65
DASH = "-" * 65


class ViTTrainer:
    """Trainer for RGB ViT Branch."""
    
    def __init__(self, cfg: Config):
        self.cfg = cfg
        
        if torch.cuda.is_available() and cfg.device == "cuda":
            self.device = torch.device("cuda")
            print(f"   Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            self.device = torch.device("cpu")
            print("  Using CPU (training will be slow)")
        
        logger.info(f"Using device: {self.device}")
        
        # ── Model ─────────────────────────────────────────────────────────────
        self.model = RGBViTBranch(cfg).to(self.device)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model: {total_params:,} total parameters, {trainable_params:,} trainable")
        
        # ── Loss ──────────────────────────────────────────────────────────────
        self.criterion = nn.BCEWithLogitsLoss()
        self.label_smoothing = cfg.label_smoothing
        
        # ── Optimizer with different LRs and weight decay ────────────────────
        param_groups = [
            {
                "params": self.model.vit.parameters(),
                "lr": cfg.lr_backbone,
                "weight_decay": cfg.weight_decay * 2,
            },
            {
                "params": self.model.classifier.parameters(),
                "lr": cfg.lr_head,
                "weight_decay": cfg.weight_decay,
            },
        ]
        
        self.optimizer = optim.AdamW(param_groups)
        
        # ── LR Schedulers ─────────────────────────────────────────────────────
        def lr_lambda(epoch: int) -> float:
            if epoch < cfg.warmup_epochs:
                return float(epoch + 1) / float(max(cfg.warmup_epochs, 1))
            return 1.0
            
        self.warmup_scheduler = LambdaLR(self.optimizer, lr_lambda=lr_lambda)
        self.cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=max(cfg.epochs - cfg.warmup_epochs, 1),
            eta_min=1e-6,
        )
        self.reduce_scheduler = ReduceLROnPlateau(
            self.optimizer, 
            mode='max', 
            factor=0.5, 
            patience=3
        )
        
        # ── Mixed precision - FIXED for PyTorch 2.5.1 ─────────────────────────
        if self.device.type == "cuda":
            self.scaler = GradScaler('cuda')
        else:
            self.scaler = None  # No scaler for CPU
        
        # ── Training state ───────────────────────────────────────────
        self.start_epoch = 0
        self.best_val_acc = 0.0
        self.best_val_auc = 0.0
        self.patience_counter = 0
        self.history: List[dict] = []
        
        # Create checkpoint directory
        self.checkpoint_dir = Path(cfg.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        print(f" Checkpoints will be saved to: {self.checkpoint_dir.absolute()}")
        print(f"    Models will be saved EVERY EPOCH as: epoch_001.pth, epoch_002.pth, etc.")
        
        # ── Auto-resume ───────────────────────────────────────────────────────
        if cfg.resume:
            latest = self.checkpoint_dir / "latest.pth"
            if cfg.resume_path:
                self._load_checkpoint(cfg.resume_path)
            elif latest.exists():
                logger.info(f"Auto-resuming from: {latest}")
                self._load_checkpoint(str(latest))
            else:
                logger.warning("No checkpoint found. Starting from scratch.")
                
    # ── Dataloaders ───────────────────────────────────────────────────────────
    
    def _build_dataloaders(self) -> tuple[DataLoader, DataLoader]:
        cfg = self.cfg
        
        print(f"\nLoading datasets...")
        print(f"   Train real: {cfg.train_real_dir}")
        print(f"   Train fake: {cfg.train_fake_dir}")
        print(f"   Val real: {cfg.val_real_dir}")
        print(f"   Val fake: {cfg.val_fake_dir}")
        
        train_dataset = DeepfakeDataset(
            real_dir=cfg.train_real_dir,
            fake_dir=cfg.train_fake_dir,
            mode="train",
            image_size=cfg.image_size,
            mean=cfg.mean,
            std=cfg.std,
        )
        
        val_dataset = DeepfakeDataset(
            real_dir=cfg.val_real_dir,
            fake_dir=cfg.val_fake_dir,
            mode="val",
            image_size=cfg.image_size,
            mean=cfg.mean,
            std=cfg.std,
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=(self.device.type == "cuda"),
            drop_last=True,
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=(self.device.type == "cuda"),
        )
        
        print(f"\nDataset sizes:")
        print(f"   Train: {len(train_dataset)} images ({len(train_loader)} batches)")
        print(f"   Val: {len(val_dataset)} images ({len(val_loader)} batches)")
        
        return train_loader, val_loader
        
    # ── Label smoothing ───────────────────────────────────────────────────────
    
    def _smooth_labels(self, labels: torch.Tensor) -> torch.Tensor:
        eps = self.label_smoothing
        return labels.float() * (1.0 - eps) + eps * 0.5
        
    # ── One training epoch - FIXED autocast ───────────────────────────────────
    
    def _train_one_epoch(self, loader: DataLoader, epoch: int) -> dict:
        self.model.train()
        
        total_loss = 0.0
        all_probs = []
        all_labels = []
        
        pbar = tqdm(loader, desc=f'Training Epoch {epoch+1}/{self.cfg.epochs}', 
                    unit='batch', leave=False, ncols=100)
        
        for batch in pbar:
            rgb = batch["rgb"].to(self.device)
            labels = batch["label"].to(self.device)
            
            labels_smooth = self._smooth_labels(labels).unsqueeze(1)
            
            # FIXED: autocast for PyTorch 2.5.1
            if self.device.type == "cuda" and self.scaler is not None:
                with autocast('cuda'):
                    logit = self.model.get_logit(rgb)
                    loss = self.criterion(logit, labels_smooth)
                
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # CPU training
                logit = self.model.get_logit(rgb)
                loss = self.criterion(logit, labels_smooth)
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                self.optimizer.step()
            
            total_loss += loss.item()
            
            with torch.no_grad():
                probs = torch.sigmoid(logit).squeeze(1)
                all_probs.extend(probs.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss/(pbar.n+1):.4f}'
            })
        
        pbar.close()
        return self._compute_metrics(all_labels, all_probs, total_loss / len(loader))
        
    # ── Validation ────────────────────────────────────────────────────────────
    
    @torch.no_grad()
    def _validate(self, loader: DataLoader) -> dict:
        self.model.eval()
        
        total_loss = 0.0
        all_probs = []
        all_labels = []
        
        pbar = tqdm(loader, desc='Validating', unit='batch', leave=False, ncols=100)
        
        for batch in pbar:
            rgb = batch["rgb"].to(self.device)
            labels = batch["label"].to(self.device)
            labels_f = labels.float().unsqueeze(1)
            
            logit = self.model.get_logit(rgb)
            loss = self.criterion(logit, labels_f)
            total_loss += loss.item()
            
            probs = torch.sigmoid(logit).squeeze(1)
            all_probs.extend(probs.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        pbar.close()
        return self._compute_metrics(all_labels, all_probs, total_loss / len(loader))
        
    # ── Metrics ───────────────────────────────────────────────────────────────
    
    def _compute_metrics(self, labels: list[int], probs: list[float], loss: float) -> dict:
        preds = [1 if p >= self.cfg.threshold else 0 for p in probs]
        labels_np = np.array(labels)
        probs_np = np.array(probs)
        
        acc = accuracy_score(labels_np, preds)
        f1 = f1_score(labels_np, preds, zero_division=0)
        
        try:
            auc = roc_auc_score(labels_np, probs_np)
        except ValueError:
            auc = 0.5
            
        return {
            "loss": round(loss, 6),
            "acc": round(acc, 6),
            "f1": round(f1, 6),
            "auc": round(auc, 6),
        }
        
    # ── Scheduler step ────────────────────────────────────────────────────────
    
    def _step_scheduler(self, epoch: int, val_acc: float):
        if epoch < self.cfg.warmup_epochs:
            self.warmup_scheduler.step()
        else:
            self.cosine_scheduler.step()
            self.reduce_scheduler.step(val_acc)
            
    # ── Save checkpoint ───────────────────────────────────────────────────────

    
    def _save_checkpoint(
        self,
        epoch: int,
        train_metrics: dict,
        val_metrics: dict,
        is_best_acc: bool,
        is_best_auc: bool,
    ):
        # Prepare checkpoint data
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_acc": self.best_val_acc,
            "best_val_auc": self.best_val_auc,
            "patience_counter": self.patience_counter,
            "history": self.history,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "cfg": self.cfg,
            "timestamp": datetime.now().isoformat(),
        }
        
        # Add scaler only if it exists
        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()
        
        # Add scheduler states
        checkpoint["warmup_sched_state"] = self.warmup_scheduler.state_dict()
        checkpoint["cosine_sched_state"] = self.cosine_scheduler.state_dict()
        checkpoint["reduce_sched_state"] = self.reduce_scheduler.state_dict()
        
        # 1. Always save latest checkpoint
        latest_path = self.checkpoint_dir / "latest.pth"
        torch.save(checkpoint, latest_path)
        
        # 2. Save best accuracy model
        if is_best_acc:
            best_acc_path = self.checkpoint_dir / "best_acc.pth"
            torch.save(checkpoint, best_acc_path)
            logger.info(f"  BEST ACCURACY MODEL SAVED: {best_acc_path}")
            logger.info(f"     Accuracy: {val_metrics['acc']:.2%}")
        
        # 3. Save best AUC model
        if is_best_auc:
            best_auc_path = self.checkpoint_dir / "best_auc.pth"
            torch.save(checkpoint, best_auc_path)
            logger.info(f"   BEST AUC MODEL SAVED: {best_auc_path}")
            logger.info(f"     AUC: {val_metrics['auc']:.4f}")
        
        # 4.  SAVE EVERY EPOCH
        epoch_path = self.checkpoint_dir / f"epoch_{epoch+1:03d}.pth"
        torch.save(checkpoint, epoch_path)
        logger.info(f"  SAVED EPOCH {epoch+1}: {epoch_path}")
        
        # 5. Save training history
        history_path = self.checkpoint_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
        
    # ── Load checkpoint ───────────────────────────────────────────────────────
    
    def _load_checkpoint(self, path: str):
        logger.info(f"Loading checkpoint: {path}")
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if "scaler_state_dict" in checkpoint and self.scaler is not None:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
        self.warmup_scheduler.load_state_dict(checkpoint["warmup_sched_state"])
        self.cosine_scheduler.load_state_dict(checkpoint["cosine_sched_state"])
        self.reduce_scheduler.load_state_dict(checkpoint["reduce_sched_state"])
        
        self.start_epoch = checkpoint["epoch"] + 1
        self.best_val_acc = checkpoint.get("best_val_acc", 0.0)
        self.best_val_auc = checkpoint.get("best_val_auc", 0.0)
        self.patience_counter = checkpoint.get("patience_counter", 0)
        self.history = checkpoint.get("history", [])
        
        logger.info(f"Resumed from epoch {self.start_epoch}")
        logger.info(f"Best Val Acc: {self.best_val_acc:.2%}")
        
    # ── Print epoch summary ───────────────────────────────────────────────────
    
    def _print_epoch_summary(
        self,
        epoch: int,
        train_metrics: dict,
        val_metrics: dict,
        is_best_acc: bool,
        is_best_auc: bool,
        epoch_time: float,
        current_lr: float,
    ):
        acc_tag = " 🏆 BEST ACC" if is_best_acc else ""
        auc_tag = " 🎯 BEST AUC" if is_best_auc else ""
        
        print(f"\n{LINE}")
        print(f"  EPOCH {epoch+1:>3} / {self.cfg.epochs} Complete")
        print(DASH)
        print(f"  {'Metric':<22}  {'TRAIN':>10}   {'VAL':>10}")
        print(DASH)
        print(
            f"  {'Loss':<22}  "
            f"{train_metrics['loss']:>10.4f}   "
            f"{val_metrics['loss']:>10.4f}"
        )
        print(
            f"  {'ACCURACY':<22}  "
            f"{train_metrics['acc']*100:>9.2f}%   "
            f"{val_metrics['acc']*100:>9.2f}%   {acc_tag}"
        )
        print(
            f"  {'AUC-ROC':<22}  "
            f"{train_metrics['auc']:>10.4f}   "
            f"{val_metrics['auc']:>10.4f}   {auc_tag}"
        )
        print(
            f"  {'F1 Score':<22}  "
            f"{train_metrics['f1']:>10.4f}   "
            f"{val_metrics['f1']:>10.4f}"
        )
        print(DASH)
        print(f"  Learning Rate      : {current_lr:.2e}")
        print(f"  Epoch time         : {epoch_time:.1f}s")
        print(f"  Best Val Acc so far: {self.best_val_acc:.2%}")
        print(f"  Models saved in    : {self.checkpoint_dir}")
        print(f"  Last saved         : epoch_{epoch+1:03d}.pth")
        print(LINE)
        
    # ── Main training loop ────────────────────────────────────────────────────
    
    def train(self):
        train_loader, val_loader = self._build_dataloaders()
        
        print(f"\n{LINE}")
        print(f"  🚀 RGB ViT-Small/16 Deepfake Detector Training")
        print(DASH)
        print(f"  Device             : {self.device}")
        print(f"  Total epochs       : {self.cfg.epochs}")
        print(f"  Starting at epoch  : {self.start_epoch + 1}")
        print(f"  Checkpoint dir     : {self.checkpoint_dir}")
        print(f"  Models will be saved EVERY EPOCH as:")
        print(f"    - epoch_001.pth, epoch_002.pth, ...")
        print(f"    - latest.pth     (resume training)")
        print(f"    - best_acc.pth   🏆 BEST MODEL FOR INFERENCE")
        print(f"    - best_auc.pth   (best AUC)")
        print(LINE)
        
        for epoch in range(self.start_epoch, self.cfg.epochs):
            t0 = time.time()
            
            train_metrics = self._train_one_epoch(train_loader, epoch)
            
            val_metrics = self._validate(val_loader)
            
            current_lr = self.optimizer.param_groups[0]['lr']
            
            self._step_scheduler(epoch, val_metrics["acc"])
            
            is_best_acc = val_metrics["acc"] > self.best_val_acc
            is_best_auc = val_metrics["auc"] > self.best_val_auc
            
            if is_best_acc:
                self.best_val_acc = val_metrics["acc"]
                self.patience_counter = 0
            elif is_best_auc:
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                
            self.history.append({
                "epoch": epoch + 1,
                "train": train_metrics,
                "val": val_metrics,
            })
            
            self._save_checkpoint(epoch, train_metrics, val_metrics, is_best_acc, is_best_auc)
            
            self._print_epoch_summary(
                epoch, train_metrics, val_metrics, is_best_acc, is_best_auc, 
                time.time() - t0, current_lr
            )
            
            # Early stopping
            if self.patience_counter >= self.cfg.early_stopping_patience:
                print(f"\n  ⏹️  Early stopping at epoch {epoch+1}")
                break
                
        print(f"\n{LINE}")
        print(f"   TRAINING COMPLETE!")
        print(f"  Best Validation Accuracy: {self.best_val_acc:.2%}")
        print(f"  Best model saved to: {self.checkpoint_dir / 'best_acc.pth'}")
        print(f"  All epochs saved as: epoch_XXX.pth")
        print(f"  Latest model: {self.checkpoint_dir / 'latest.pth'}")
        print(LINE)
        self._print_full_history()
        
    # ── Full history table ────────────────────────────────────────────────────
    
    def _print_full_history(self):
        print(f"\n{LINE}")
        print(f"  📊 FULL TRAINING HISTORY")
        print(DASH)
        print(
            f"  {'Epoch':>5}  "
            f"{'T-Loss':>8}  {'T-Acc':>7}  {'T-AUC':>7}  "
            f"{'V-Loss':>8}  {'V-Acc':>7}  {'V-AUC':>7}"
        )
        print(DASH)
        for row in self.history:
            t = row["train"]
            v = row["val"]
            best_marker = "🏆" if v["acc"] == self.best_val_acc else " "
            print(
                f"  {row['epoch']:>5}  "
                f"{t['loss']:>8.4f}  {t['acc']*100:>6.2f}%  {t['auc']:>7.4f}  "
                f"{v['loss']:>8.4f}  {v['acc']*100:>6.2f}%{best_marker}  {v['auc']:>7.4f}"
            )
        print(LINE)
# finetune.py
import os
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np

from config import cfg
from model import iresnet50, ArcMarginProduct
from utils import get_transforms, get_dataloaders

def finetune():
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    train_transform, val_transform = get_transforms()
    train_loader, val_loader, class_names = get_dataloaders(cfg.DATA_ROOT, train_transform, val_transform)
    num_classes = len(class_names)
    print(f"Fine-tuning for {num_classes} identities.")

    # 1. Model & Head
    model = iresnet50(fp16=False).to(cfg.DEVICE)
    arcface_head = ArcMarginProduct(cfg.EMBEDDING_SIZE, num_classes, s=cfg.ARCFACE_S, m=cfg.ARCFACE_M).to(cfg.DEVICE)

    # 2. Load pre-trained weights
    try:
        model.load_state_dict(torch.load(cfg.PRETRAINED_RECOGNITION_MODEL_PATH, map_location=cfg.DEVICE, weights_only=True), strict=False)
        print(f"Successfully loaded pre-trained weights from {cfg.PRETRAINED_RECOGNITION_MODEL_PATH}")
    except FileNotFoundError:
        print(f"WARNING: Pre-trained model not found at {cfg.PRETRAINED_RECOGNITION_MODEL_PATH}. Training from scratch.")

    # 3. Optimizer, Criterion & Scheduler
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(arcface_head.parameters()), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.STEP_SIZE, gamma=cfg.GAMMA)
    
    # Biến để theo dõi loss tốt nhất trên tập validation
    best_val_loss = np.inf
    
    # Biến đếm cho Early Stopping
    patience_counter = 0
    
    # 4. Fine-tuning Loop
    for epoch in range(1, cfg.EPOCHS + 1):
        model.train()
        arcface_head.train()
        
        epoch_train_loss = 0.0
        
        # Lấy learning rate hiện tại để hiển thị
        current_lr = scheduler.get_last_lr()[0]
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.EPOCHS} | LR: {current_lr:.1e}")
        
        for imgs, labels in pbar:
            imgs, labels = imgs.to(cfg.DEVICE), labels.to(cfg.DEVICE)
            
            emb = model(imgs)
            logits = arcface_head(emb, labels)
            loss = criterion(logits, labels)
            
            epoch_train_loss += loss.item() # <-- Thêm loss của batch vào
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        # In ra loss trung bình của epoch
        avg_epoch_loss = epoch_train_loss / len(train_loader)
        print(f"Epoch {epoch}/{cfg.EPOCHS} - Average Loss: {avg_epoch_loss:.4f}")
        
        scheduler.step()

        # --- VALIDATION PHASE ---
        if val_loader is not None:
            model.eval()
            arcface_head.eval()
            epoch_val_loss = 0.0
            
            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{cfg.EPOCHS} | Validation")
                for imgs, labels in val_pbar:
                    imgs, labels = imgs.to(cfg.DEVICE), labels.to(cfg.DEVICE)
                    
                    emb = model(imgs)
                    logits = arcface_head(emb, labels)
                    loss = criterion(logits, labels)
                    
                    epoch_val_loss += loss.item()
                    val_pbar.set_postfix(loss=f"{loss.item():.4f}")

            avg_val_loss = epoch_val_loss / len(val_loader)
            print(f"Epoch {epoch}/{cfg.EPOCHS} - Average Validation Loss: {avg_val_loss:.4f}")

            # --- MODEL CHECKPOINTING & EARLY STOPPING LOGIC ---
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                print(f"✅ New best model found! Saving to {cfg.FINETUNED_MODEL_PATH}")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'arcface_head_state_dict': arcface_head.state_dict(),
                    'class_names': class_names,
                    'val_loss': best_val_loss
                }, cfg.FINETUNED_MODEL_PATH)
                
                # Reset bộ đếm kiên nhẫn khi tìm thấy model tốt hơn
                patience_counter = 0
            else:
                # Nếu không tốt hơn, tăng bộ đếm
                patience_counter += 1
                print(f"⚠️ No improvement in val_loss. Patience: {patience_counter} / {cfg.EARLY_STOPPING_PATIENCE}")

            # Kiểm tra điều kiện dừng sớm
            if patience_counter >= cfg.EARLY_STOPPING_PATIENCE:
                print(f"❌ Early Stopping triggered after {epoch} epochs. Best val_loss: {best_val_loss:.4f}")
                break

    print("Fine-tuning finished.")
    # Lưu ý: cfg.FINETUNED_MODEL_PATH giờ đây sẽ chứa model tốt nhất dựa trên tập val

if __name__ == "__main__":
    finetune()
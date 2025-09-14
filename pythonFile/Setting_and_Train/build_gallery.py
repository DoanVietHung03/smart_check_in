# build_gallery.py
import os
import torch
import numpy as np
import faiss
from tqdm import tqdm
from PIL import Image
from torchvision import datasets
import json

from config import cfg
from model import iresnet34
from utils import FaceDetector, align_face, get_transforms

def build_gallery():
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    # 1. Load model đã fine-tuning
    model = iresnet34(fp16=False).to(cfg.DEVICE)
    ckpt = torch.load(cfg.FINETUNED_MODEL_PATH, map_location=cfg.DEVICE, weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    class_names = ckpt['class_names']
    print("Fine-tuned model loaded.")

    # 2. Khởi tạo detector và transform
    detector = FaceDetector()
    _, transform = get_transforms()

    # 3. Trích xuất embedding cho gallery
    all_feats = []
    all_labels = []
    
    train_dir = os.path.join(cfg.DATA_ROOT, "train")
    dataset = datasets.ImageFolder(train_dir)
    
    ALIGNED_DEBUG_DIR = os.path.join("..", "aligned_debug")
    os.makedirs(ALIGNED_DEBUG_DIR, exist_ok=True)
    print(f"Saving aligned faces for debugging in: {ALIGNED_DEBUG_DIR}")
    
    print("Building gallery... This may take a while.")
    for path, label_idx in tqdm(dataset.imgs):
        try:
            image_np = np.array(Image.open(path).convert("RGB"))
            boxes, landmarks = detector.detect(image_np)

            if len(boxes) == 1:
                aligned_face = align_face(image_np, landmarks[0], boxes[0])
                
                # >> THÊM VÀO: Lưu ảnh đã căn chỉnh
                person_name = class_names[label_idx]
                person_debug_dir = os.path.join(ALIGNED_DEBUG_DIR, person_name)
                os.makedirs(person_debug_dir, exist_ok=True)
                
                img_to_save = Image.fromarray(aligned_face)
                img_to_save.save(os.path.join(person_debug_dir, os.path.basename(path)))
                
                face_tensor = transform(Image.fromarray(aligned_face)).unsqueeze(0).to(cfg.DEVICE)
                
                with torch.no_grad():
                    feat = model(face_tensor).cpu().numpy()
                
                all_feats.append(feat)
                all_labels.append(label_idx)
        except Exception as e:
            print(f"Skipping {path} due to error: {e}")

    # 4. Tính embedding trung bình
    gallery_feats = []
    gallery_labels = []
    all_feats = np.vstack(all_feats)
    all_labels = np.array(all_labels)

    for lab_idx in sorted(set(all_labels)):
        gallery_feats.append(all_feats[all_labels == lab_idx].mean(axis=0))
        gallery_labels.append(lab_idx)

    gallery_feats = np.array(gallery_feats, dtype='float32')
    faiss.normalize_L2(gallery_feats) # Chuẩn hóa L2 cho IndexFlatIP
    
    # 5. Build FAISS index
    index = faiss.IndexFlatIP(cfg.EMBEDDING_SIZE)
    index.add(gallery_feats)
    faiss.write_index(index, cfg.FAISS_INDEX_PATH)
    print(f"FAISS index with {index.ntotal} identities saved to {cfg.FAISS_INDEX_PATH}")

    # 6. Lưu id2name và name2path mappings
    id2name = {}
    name2path = {}
    for faiss_idx, lab_idx in enumerate(gallery_labels):
        name = class_names[lab_idx]
        id2name[faiss_idx] = name
        
        if name not in name2path:
            for path, label_index in dataset.imgs:
                if label_index == lab_idx:
                    name2path[name] = path
                    break

    with open(cfg.ID2NAME_PATH, "w", encoding="utf-8") as f:
        for idx, name in id2name.items():
            f.write(f"{idx}\t{name}\n")
    print(f"ID-to-Name mapping saved to {cfg.ID2NAME_PATH}")

    with open(cfg.NAME2PATH_PATH, "w", encoding="utf-8") as f:
        json.dump(name2path, f, ensure_ascii=False, indent=4)
    print(f"Name-to-Path mapping saved to {cfg.NAME2PATH_PATH}")
    
    # 7. Save embeddings & labels for visualization
    np.save(os.path.join(cfg.OUTPUT_DIR, "gallery_feats.npy"), gallery_feats)
    np.save(os.path.join(cfg.OUTPUT_DIR, "gallery_labels.npy"), gallery_labels)
    print("Saved gallery_feats.npy and gallery_labels.npy for visualization.")

if __name__ == "__main__":
    build_gallery()
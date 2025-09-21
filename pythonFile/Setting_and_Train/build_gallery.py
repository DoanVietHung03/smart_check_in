# build_gallery.py
import os
import torch
import numpy as np
import faiss
from tqdm import tqdm
from PIL import Image, ImageFilter
from torchvision import datasets, transforms
import json
import sys

from config import cfg
from model import iresnet34
from utils import FaceDetector_YOLO, FaceDetector_RetinaFace, align_face, get_transforms

def build_gallery():
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    # 1. Load model nhận diện khuôn mặt
    model = iresnet34(fp16=False).to(cfg.DEVICE)
    try:
        ckpt = torch.load(cfg.PRETRAINED_RECOGNITION_MODEL_PATH, map_location=cfg.DEVICE)
        model.load_state_dict(ckpt, strict=True)
        print(f"Successfully loaded FINE-TUNED weights from: {cfg.PRETRAINED_RECOGNITION_MODEL_PATH}")
    except Exception as e:
        print(f"FATAL: Could not load pre-trained weights. Error: {e}")
        return

    model.eval()
    # -----------------------------------------------

    # 2. Khởi tạo detector và transform
    detector = FaceDetector_RetinaFace()
    _, transform = get_transforms()

    # 3. Trích xuất embedding cho gallery
    all_feats = []
    all_labels = []
    
    train_dir = cfg.DATA_ROOT
    dataset = datasets.ImageFolder(train_dir)
    
    # Lấy class_names trực tiếp từ dataset (đáng tin cậy hơn là từ checkpoint)
    class_names = dataset.classes
    if not class_names:
        print(f"FATAL: Không tìm thấy thư mục danh tính nào trong: {train_dir}")
        return
    print(f"Found {len(class_names)} identities. Building gallery...")

    ALIGNED_DEBUG_DIR = os.path.join(cfg.OUTPUT_DIR, "aligned_debug") # Sửa đường dẫn cho đúng
    os.makedirs(ALIGNED_DEBUG_DIR, exist_ok=True)
    print(f"Saving aligned faces for debugging in: {ALIGNED_DEBUG_DIR}")
    
    for path, label_idx in tqdm(dataset.imgs):
        try:
            image_np = np.array(Image.open(path).convert("RGB"))
            boxes, landmarks = detector.detect(image_np)

            if len(boxes) >= 1:
                # Chỉ lấy khuôn mặt lớn nhất (đầu tiên) nếu có nhiều
                aligned_face = align_face(image_np, landmarks[0], boxes[0])
                aligned_pil = Image.fromarray(aligned_face) # Chuyển sang PIL Image để transform

                # Định nghĩa các phép biến đổi nhẹ
                augment_transforms = transforms.Compose([
                    transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2),
                ])
    
                try:
                    # 1. Xử lý ảnh gốc
                    face_tensor_orig = transform(Image.fromarray(aligned_face)).unsqueeze(0).to(cfg.DEVICE)
                    with torch.no_grad():
                        feat_orig = model(face_tensor_orig).cpu().detach().numpy()
                    
                    all_feats.append(feat_orig)
                    all_labels.append(label_idx)

                    # 2. Xử lý ảnh lật (Flipped)
                    aligned_face_flipped = transforms.functional.hflip(aligned_pil)
                    face_tensor_flipped = transform(aligned_face_flipped).unsqueeze(0).to(cfg.DEVICE)
                    with torch.no_grad():
                        feat_flipped = model(face_tensor_flipped).cpu().detach().numpy()
                    
                    all_feats.append(feat_flipped)
                    all_labels.append(label_idx) # Vẫn là label đóng góp bởi cùng một người
                    
                    for _ in range(8):
                        augmented_pil = augment_transforms(aligned_pil)
                        feat_aug = model(transform(augmented_pil).unsqueeze(0).to(cfg.DEVICE)).cpu().detach().numpy()
                        all_feats.append(feat_aug)
                        all_labels.append(label_idx)
                    
                    # Lưu ảnh đã căn chỉnh
                    person_name = class_names[label_idx]
                    person_debug_dir = os.path.join(ALIGNED_DEBUG_DIR, person_name)
                    os.makedirs(person_debug_dir, exist_ok=True)
                    img_to_save = Image.fromarray(aligned_face)
                    img_to_save.save(os.path.join(person_debug_dir, os.path.basename(path)))
                except Exception as e:
                    print(f"Skipping {path} due to error: {e}")
            else:
                 print(f"Warning: No face detected in {path}. Skipping.")
        except Exception as e:
            print(f"Skipping {path} due to error: {e}")

    if not all_feats:
        print("FATAL: No features extracted. Cannot build gallery. Did you add photos to Dataset_Recognition/train/?")
        return
    
    # Sử dụng TẤT CẢ các feats. 
    gallery_feats = np.vstack(all_feats) # Chuyển list các array (1, 512) thành 1 array (N, 512)
    gallery_labels = np.array(all_labels) # Chuyển list các label_idx thành 1 array (N,)
    
    gallery_feats = np.array(gallery_feats, dtype='float32')
    faiss.normalize_L2(gallery_feats) # Chuẩn hóa L2
    
    # 5. Build FAISS index
    index = faiss.IndexFlatIP(cfg.EMBEDDING_SIZE) # IP = Inner Product (Cosine Similarity)
    index.add(gallery_feats)
    faiss.write_index(index, cfg.FAISS_INDEX_PATH)
    # Lưu ý: index.ntotal bây giờ là TỔNG SỐ ẢNH, không phải tổng số người
    print(f"FAISS index with {index.ntotal} TOTAL IMAGES saved to {cfg.FAISS_INDEX_PATH}") 

    # 6. Lưu bản đồ (mappings)
    id2name = {} # Map này bây giờ sẽ map: FAISS_Index -> Tên
    name2path = {}
    
    # Lấy đường dẫn gốc của gallery để tính toán đường dẫn tương đối
    data_root_path = os.path.abspath(cfg.DATA_ROOT)
    
    for faiss_idx, lab_idx in enumerate(gallery_labels):
        name = class_names[lab_idx]
        id2name[faiss_idx] = name # Ví dụ: {0: "John", 1: "John", 2: "Jane", ...}
        
        # Tìm MỘT ảnh đại diện cho người này để lưu vào name2path
        if name not in name2path:
            for path, label_index in dataset.imgs:
                if label_index == lab_idx:
                    # 1. Lấy đường dẫn tương đối
                    rel_path = os.path.relpath(path, data_root_path)
                    
                    # 2. Chuyển đổi sang định dạng URL (dùng dấu /)
                    web_path = rel_path.replace(os.path.sep, '/')
                    
                    # 3. Lưu đường dẫn web tương đối
                    name2path[name] = web_path
                    break

    with open(cfg.ID2NAME_PATH, "w", encoding="utf-8") as f:
        json.dump(id2name, f, ensure_ascii=False, indent=4)
    print(f"ID-to-Name (Index-to-Name) mapping saved to {cfg.ID2NAME_PATH}")

    with open(cfg.NAME2PATH_PATH, "w", encoding="utf-8") as f:
        json.dump(name2path, f, ensure_ascii=False, indent=4)
    print(f"Name-to-Path mapping saved to {cfg.NAME2PATH_PATH}")
    
    # 7. Lưu feats để trực quan hóa (optional)
    np.save(os.path.join(cfg.OUTPUT_DIR, "gallery_feats.npy"), gallery_feats)
    np.save(os.path.join(cfg.OUTPUT_DIR, "gallery_labels.npy"), gallery_labels)
    print("Saved gallery features for visualization.")

if __name__ == "__main__":
    build_gallery()
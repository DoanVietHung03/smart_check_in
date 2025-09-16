import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import sys 

# --- Thêm đường dẫn đến thư mục Setting&Train ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  
setting_and_train_dir = os.path.join(project_root, "Setting_and_Train")
sys.path.append(setting_and_train_dir)
# --------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import json
from torchvision import datasets

from config import cfg

# ================ THAY ĐỔI Ở ĐÂY ================
# 1. Load embeddings & labels đã lưu từ extract_features.py
feats_path = os.path.join(cfg.OUTPUT_DIR, "gallery_feats.npy")
labels_path = os.path.join(cfg.OUTPUT_DIR, "gallery_labels.npy")
# ===============================================

if not os.path.exists(feats_path) or not os.path.exists(labels_path):
    raise FileNotFoundError("Chưa có file .npy. Hãy chạy extract_features.py trước.")

embs = np.load(feats_path)
labels = np.load(labels_path)

print(f"Loaded {len(embs)} embeddings with dim {embs.shape[1]}.")

# 2. Lấy id2name từ ImageFolder (không cần file id2name.txt nữa)
train_dir = os.path.join(cfg.DATA_ROOT, "train")
dataset = datasets.ImageFolder(train_dir)
id2name = {i: name for i, name in enumerate(dataset.classes)}
print("Mapping id2name:", id2name)

# 3. Tự động chọn perplexity
n_samples = len(embs)
if n_samples <= 1:
    raise ValueError("Không đủ số mẫu để chạy t-SNE (cần > 3).")

# Giảm perplexity vì số mẫu rất nhỏ
perplexity = max(2, min(30, n_samples - 2)) 
print(f"Using perplexity={perplexity} for n_samples={n_samples}")

# 4. Giảm chiều bằng t-SNE
tsne = TSNE(
    n_components=2,
    perplexity=perplexity,
    learning_rate='auto', # 'auto' thường tốt hơn
    init='pca',
    random_state=42
)
reduced = tsne.fit_transform(embs)

# 5. Vẽ scatter plot
plt.figure(figsize=(10, 8))
unique_labels = np.unique(labels)

for lab in unique_labels:
    mask = labels == lab
    name = id2name.get(int(lab), f"ID {lab}")
    plt.scatter(reduced[mask, 0], reduced[mask, 1], label=name, alpha=0.7, s=100)

plt.legend()
plt.title("t-SNE Visualization of Backbone Embeddings")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.grid(True)
plt.show() 
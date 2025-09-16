# config.py
import torch
import os

class Config:
    # --- Paths ---
    BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Lấy thư mục gốc của dự án
    DATA_ROOT = os.path.join(BASE_DIR, "..", "..", "Gallery")
    WEIGHTS_DIR = os.path.join(BASE_DIR, "..", "..", "pretrained_model_weights")
    OUTPUT_DIR = os.path.join(BASE_DIR, "..", "..", "checkpoints")
    
    # Các đường dẫn khác sẽ tự động đúng
    DETECTOR_MODEL_PATH = os.path.join(WEIGHTS_DIR, "Face_detection_yolo", "yolov11n-face.pt")
    PRETRAINED_RECOGNITION_MODEL_PATH = os.path.join(WEIGHTS_DIR, "Face_recognition_iResNet", "arcFace_r34.pth")

    FAISS_INDEX_PATH = os.path.join(OUTPUT_DIR, "faiss_gallery.idx")
    ID2NAME_PATH = os.path.join(OUTPUT_DIR, "id2name.txt")
    NAME2PATH_PATH = os.path.join(OUTPUT_DIR, "name2path.json")

    # --- Data & Model ---
    IMG_SIZE = (112, 112)
    EMBEDDING_SIZE = 512
    
    # --- Training & Fine-tuning ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 32
    EPOCHS = 100 
    LR = 1e-4
    WEIGHT_DECAY = 5e-4
    STEP_SIZE = 5     
    GAMMA = 0.1   
    EARLY_STOPPING_PATIENCE = 10    

    # --- ArcFace ---
    ARCFACE_S = 64.0
    ARCFACE_M = 0.50

    # --- Inference & Real-time ---
    TOP_K = 5 
    DETECTION_CONFIDENCE = 0.8 
    RECOGNITION_THRESHOLD = 0.75 

cfg = Config()
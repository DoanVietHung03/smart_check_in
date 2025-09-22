# config.py
import torch
import os

class Config:
    # --- Lựa chọn Model Detector ---
    # Thay đổi giá trị này thành 'yolo' hoặc 'retinaface'
    DETECTOR_TYPE = 'retinaface'
    
    # --- Paths ---
    BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Lấy thư mục gốc của dự án
    DATA_ROOT = os.path.join(BASE_DIR, "..", "..", "Gallery")
    WEIGHTS_DIR = os.path.join(BASE_DIR, "..", "..", "pretrained_model_weights")
    OUTPUT_DIR = os.path.join(BASE_DIR, "..", "..", "checkpoints")
    
    # Các đường dẫn khác sẽ tự động đúng
    DETECTOR_MODEL_PATH = os.path.join(WEIGHTS_DIR, "Face_detection_yolo", "yolov11n-face.pt") # <---- Đường dẫn này chỉ được sử dụng nếu DETECTOR_TYPE = 'yolo'
    PRETRAINED_RECOGNITION_MODEL_PATH = os.path.join(WEIGHTS_DIR, "Face_recognition_iresnet", "arcFace_r34.pth")
    BEST_FINETUNED_MODEL_PATH = os.path.join(OUTPUT_DIR, "best_finetuned_model.pth")

    FAISS_INDEX_PATH = os.path.join(OUTPUT_DIR, "faiss_gallery.idx")
    ID2NAME_PATH = os.path.join(OUTPUT_DIR, "id2name.json")
    NAME2PATH_PATH = os.path.join(OUTPUT_DIR, "name2path.json")

    # --- Data & Model ---
    IMG_SIZE = (112, 112)
    EMBEDDING_SIZE = 512
    
    # --- Training & Fine-tuning ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 16
    EPOCHS = 120 
    LR = 1e-4
    WEIGHT_DECAY = 5e-4
    STEP_SIZE = 30     
    GAMMA = 0.1   
    EARLY_STOPPING_PATIENCE = 30    

    # --- ArcFace ---
    ARCFACE_S = 64.0
    ARCFACE_M = 0.50

    # --- Inference & Real-time ---
    TOP_K = 5 
    DETECTION_CONFIDENCE = 0.7 
    RECOGNITION_THRESHOLD = 0.5

cfg = Config()
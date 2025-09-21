import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import os
import torch
import json
import numpy as np
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import cv2
from retinaface import RetinaFace   
from ultralytics import YOLO
from skimage import transform as trans

from config import cfg

class FaceDetector_YOLO:
    def __init__(self, model_path=cfg.DETECTOR_MODEL_PATH):
        self.model = YOLO(model_path)

    def detect(self, image_np):
        results = self.model(image_np, device=cfg.DEVICE, verbose=False)
        boxes = []
        landmarks = []
        for res in results:
            if res.boxes is None: continue
            xyxyn = res.boxes.xyxyn.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            
            if res.keypoints is not None and len(res.keypoints.xy) > 0:
                keypoints_xy = res.keypoints.xy.cpu().numpy()
            else: 
                keypoints_xy = np.zeros((len(xyxyn), 5, 2))

            for i in range(len(xyxyn)):
                if confs[i] > cfg.DETECTION_CONFIDENCE:
                    h, w, _ = image_np.shape
                    x1, y1, x2, y2 = xyxyn[i]
                    box = [int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)]
                    boxes.append(box)
                    landmarks.append(keypoints_xy[i])

        return boxes, landmarks

class FaceDetector_RetinaFace:
    def __init__(self):
        # Thư viện RetinaFace tự động xử lý việc tải model
        # nên chúng ta không cần truyền đường dẫn vào nữa.
        print("Initialized RetinaFace detector.")

    def detect(self, image_np):
        """
        Phát hiện khuôn mặt bằng RetinaFace và trả về boxes, landmarks
        theo đúng định dạng mà hệ thống đang cần.
        """
        boxes = []
        landmarks = []
        
        try:
            # RetinaFace trả về một dictionary, với mỗi key là một khuôn mặt (ví dụ: 'face_1')
            # Chúng ta đặt threshold ngay trong lệnh gọi để lọc bớt các phát hiện nhiễu
            results = RetinaFace.detect_faces(image_np, threshold=cfg.DETECTION_CONFIDENCE)
            
            if isinstance(results, dict):
                for face_name, face_info in results.items():
                    # 1. Lấy bounding box
                    # Định dạng của 'facial_area' là [x1, y1, x2, y2], đúng như ta cần
                    box = face_info['facial_area']
                    # Chuyển đổi sang int để tương thích
                    boxes.append([int(b) for b in box])

                    # 2. Lấy landmarks theo đúng thứ tự cho hàm align_face
                    lm = face_info['landmarks']
                    landmark_points = [
                        lm['right_eye'],
                        lm['left_eye'],
                        lm['nose'],
                        lm['mouth_right'],
                        lm['mouth_left']
                    ]
                    landmarks.append(np.array(landmark_points, dtype=np.float32))

        except Exception as e:
            # Nếu không tìm thấy khuôn mặt, thư viện có thể báo lỗi hoặc trả về rỗng.
            # Trả về list rỗng là hành vi đúng đắn.
            pass

        return boxes, landmarks

def align_face(image_np, landmarks, box=None):
    # Đường dẫn chính: Dùng landmark để biến đổi
    if landmarks is not None and np.any(landmarks != 0):
        src = np.array([
            [38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
            [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)

        dst = np.array(landmarks, dtype=np.float32).reshape(5, 2)
        
        tform = trans.SimilarityTransform()
        tform.estimate(dst, src)
        M = tform.params[0:2, :]
        
        aligned = cv2.warpAffine(image_np, M, (cfg.IMG_SIZE[1], cfg.IMG_SIZE[0]), borderValue=0.0)
        return aligned

    # Đường dẫn phụ (Fallback): Dùng bounding box (VỚI KIỂM TRA AN TOÀN)
    if box is not None:
        h, w, _ = image_np.shape  # Lấy kích thước ảnh gốc
        x1, y1, x2, y2 = [int(b) for b in box]
        
        # --- KIỂM TRA AN TOÀN ---
        # 1. Kẹp tọa độ để đảm bảo nó LUÔN NẰM TRONG khung hình
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)  # w là chiều rộng (cols)
        y2 = min(h, y2)  # h là chiều cao (rows)

        # 2. Kiểm tra xem box có diện tích hợp lệ không (rộng > 0 và cao > 0)
        if x1 >= x2 or y1 >= y2:
            # Nếu box không hợp lệ, ném ra lỗi để bỏ qua
            raise ValueError(f"Invalid bounding box after clipping: {[x1, y1, x2, y2]}")
        
        # Nếu ổn, tiến hành crop
        cropped = image_np[y1:y2, x1:x2]
        return cv2.resize(cropped, cfg.IMG_SIZE)
    
    # Fallback cuối cùng nếu không có cả landmark và box
    return cv2.resize(image_np, cfg.IMG_SIZE)

def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize(cfg.IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.GaussianBlur(kernel_size=(3, 3)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(cfg.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])  
    return train_transform, val_transform

def get_dataloaders(data_root, train_transform, val_transform):
    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")
    
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=4)
    
    val_loader = None
    if os.path.exists(val_dir):
        val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
        val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=4)

    class_names = train_dataset.classes
    return train_loader, val_loader, class_names

def load_id2name(path):
    with open(cfg.ID2NAME_PATH, "r", encoding="utf-8") as f:
        id2name = json.load(f)
    # convert key về int
    id2name = {int(k): v for k, v in id2name.items()}
    return id2name
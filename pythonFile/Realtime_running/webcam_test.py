# webcam_test.py (PHIÊN BẢN NÂNG CẤP - BẤT ĐỒNG BỘ)
import cv2
import torch
import faiss
import numpy as np
import time
from PIL import Image
import sys
import os
import threading
import queue

# Thêm đường dẫn để import các module cần thiết
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  
setting_and_train_dir = os.path.join(project_root, "Setting_and_Train")
sys.path.append(setting_and_train_dir)

from config import cfg
from model import iresnet34
from utils import FaceDetector_YOLO, FaceDetector_RetinaFace, align_face, get_transforms, load_id2name
from supervision import Detections, ByteTrack

# === CẤU HÌNH ===
WEBCAM_ID = 0 

# Hàng đợi để giao tiếp giữa các luồng
recognition_queue = queue.Queue()
results_queue = queue.Queue()

# ===================================================================
# LUỒNG PHỤ: CHUYÊN XỬ LÝ AI
# ===================================================================
def recognition_worker(device, recognizer_model, faiss_index, id2name, recog_transform):
    print("[Worker Thread] Recognition worker started.")
    while True:
        item = recognition_queue.get()
        if item is None:  # Tín hiệu để dừng
            break
        
        frame_rgb, faces_to_process = item
        results = {}

        batch_tensors = []
        valid_tracker_ids = []
        for tracker_id, landmark, xyxy in faces_to_process:
            try:
                aligned_face = align_face(frame_rgb, landmark, xyxy)
                face_tensor = recog_transform(Image.fromarray(aligned_face))
                batch_tensors.append(face_tensor)
                valid_tracker_ids.append(tracker_id)
            except Exception:
                results[tracker_id] = ("Error", 0.0)

        if batch_tensors:
            batch = torch.stack(batch_tensors).to(device)
            with torch.no_grad():
                feats_batch = recognizer_model(batch).cpu().numpy()
            
            faiss.normalize_L2(feats_batch)
            scores, indices = faiss_index.search(feats_batch, 1)

            for i, tracker_id in enumerate(valid_tracker_ids):
                rec_score = scores[i][0]
                rec_idx = indices[i][0]
                name = "Unknown"
                if rec_score > cfg.RECOGNITION_THRESHOLD:
                    name = id2name.get(rec_idx, "Unknown")
                results[tracker_id] = (name, rec_score)
        
        if results:
            results_queue.put(results)

        recognition_queue.task_done()

# ===================================================================
# LUỒNG CHÍNH: WEBCAM VÀ HIỂN THỊ
# ===================================================================

# === CẤU HÌNH ===
WEBCAM_ID = 0 
CACHE_TTL_SECONDS = 5.0

def main():
    # --- 1. TẢI MODEL VÀ DỮ LIỆU ---
    print("[Main Thread] Loading models and gallery data...")
    device = torch.device(cfg.DEVICE)
    detector = FaceDetector_RetinaFace()
    recognizer_model = iresnet34(fp16=False).to(device)
    try:
        ckpt = torch.load(cfg.PRETRAINED_RECOGNITION_MODEL_PATH, map_location=device)
        recognizer_model.load_state_dict(ckpt, strict=True)
        recognizer_model.eval()
    except Exception as e:
        print(f"FATAL: Could not load recognizer model. Error: {e}")
        return
    faiss_index = faiss.read_index(cfg.FAISS_INDEX_PATH)
    id2name = load_id2name(cfg.ID2NAME_PATH)
    _, recog_transform = get_transforms()
    tracker = ByteTrack(frame_rate=30)
    rng = np.random.default_rng(42)
    name2color = {name: tuple(int(c) for c in rng.integers(0, 255, size=3)) for name in set(id2name.values())}
    name2color.update({"Unknown": (0, 0, 255), "Processing...": (192, 192, 192), "Error": (0, 255, 255)})
    
    tracker_results_cache = {}

    # --- 2. KHỞI ĐỘNG WORKER THREAD ---
    worker_thread = threading.Thread(
        target=recognition_worker,
        args=(device, recognizer_model, faiss_index, id2name, recog_transform),
        daemon=True
    )
    worker_thread.start()

    # --- 3. MỞ WEBCAM VÀ VÒNG LẶP CHÍNH ---
    cap = cv2.VideoCapture(WEBCAM_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print(f"FATAL: Cannot open webcam with ID {WEBCAM_ID}.")
        recognition_queue.put(None)
        worker_thread.join()
        return

    print("\nStarting webcam feed. Press 'q' to quit.")
    frame_count = 0
    tracked_detections = Detections.empty()

    while True:
        ret, frame = cap.read()
        if not ret: break

        current_time = time.time()

        # --- CẬP NHẬT KẾT QUẢ TỪ WORKER ---
        try:
            new_results = results_queue.get_nowait()
            for tracker_id, (name, score) in new_results.items():
                tracker_results_cache[tracker_id] = (name, score, current_time)
        except queue.Empty:
            pass

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        if frame_count % 2 == 0:
            boxes, landmarks = detector.detect(frame_rgb)
            
            # =========================================================
            # === SỬA LỖI Ở ĐÂY: Chỉ xử lý nếu phát hiện thấy khuôn mặt ===
            if len(boxes) > 0:
                detections_sv = Detections(
                    xyxy=np.array(boxes, dtype=float),
                    confidence=np.ones(len(boxes)),
                    data={'landmarks': landmarks}
                )
                tracked_detections = tracker.update_with_detections(detections_sv)
                
                faces_to_process = []
                for i in range(len(tracked_detections)):
                    tracker_id = int(tracked_detections.tracker_id[i])
                    
                    needs_recognition = False
                    cached_result = tracker_results_cache.get(tracker_id)

                    if cached_result is None:
                        needs_recognition = True
                    else:
                        _, _, timestamp = cached_result
                        if current_time - timestamp > CACHE_TTL_SECONDS:
                            needs_recognition = True
                    
                    if needs_recognition:
                        faces_to_process.append((
                            tracker_id,
                            tracked_detections.data['landmarks'][i],
                            tracked_detections.xyxy[i]
                        ))
                
                if faces_to_process and recognition_queue.empty():
                    recognition_queue.put((frame_rgb.copy(), faces_to_process))
            else:
                # Nếu không có khuôn mặt nào, cập nhật tracker với một đối tượng rỗng
                tracked_detections = tracker.update_with_detections(Detections.empty())
            # =========================================================
        
        # --- VẼ KẾT QUẢ ---
        for i in range(len(tracked_detections)):
            xyxy = tracked_detections.xyxy[i]
            tracker_id = int(tracked_detections.tracker_id[i])
            name, score, _ = tracker_results_cache.get(tracker_id, ("Processing...", 0.0, 0))
            
            label = f"{name} ({score:.2f})"
            color_bgr = name2color.get(name, (0, 0, 255))
            x1, y1, x2, y2 = [int(c) for c in xyxy]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, 2)
            text_y = y1 - 10 if y1 - 10 > 10 else y2 + 20
            cv2.putText(frame, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_bgr, 2)

        cv2.imshow('Webcam Face Recognition (Press "q" to quit)', frame)
        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # --- 4. DỌN DẸP ---
    print("Shutting down...")
    recognition_queue.put(None)
    worker_thread.join()
    cap.release()
    cv2.destroyAllWindows()
    print("Program finished.")

if __name__ == "__main__":
    main()
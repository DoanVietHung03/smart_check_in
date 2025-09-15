# server.py
import sys
import os
import cv2
import torch
import faiss
import numpy as np
from PIL import Image
import time
import threading
import queue
import json
import base64
import asyncio
import sqlite3
from datetime import datetime
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import logging

# Thiết lập logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI & Uvicorn
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.concurrency import run_in_threadpool
from starlette.websockets import WebSocketState

# Thư viện Detection & Tracking
from supervision import ByteTrack, Detections
from ultralytics import YOLO

# Hàm khởi tạo DB
from .db_manager import db_pool, initialize_database

# --- Thêm đường dẫn và import các module ML ở folder Setting&Train ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  
setting_and_train_dir = os.path.join(project_root, "Setting_and_Train")
sys.path.append(setting_and_train_dir)

from config import cfg
from model import iresnet34
from utils import align_face, get_transforms, load_id2name

logger.info("Imports completed. Loading models...")

# ==========================================================
# 1. TẢI MODEL TOÀN CỤC (GLOBAL MODELS) VÀ CÁC THAM SỐ KHÁC 
# ==========================================================
DEVICE = torch.device(cfg.DEVICE)

# 1.1. Detector
DETECTOR_MODEL = YOLO(cfg.DETECTOR_MODEL_PATH).to(DEVICE)
DETECTOR_MODEL.fuse()
logger.info("Global Face Detector (YOLO) loaded.")

# 1.2. Recognizer
RECOGNIZER_MODEL = iresnet34(fp16=False).to(DEVICE)
ckpt = torch.load(cfg.FINETUNED_MODEL_PATH, map_location=DEVICE, weights_only=True)
RECOGNIZER_MODEL.load_state_dict(ckpt['model_state_dict'], strict=False)
RECOGNIZER_MODEL.eval()
logger.info("Global Face Recognizer (iResNet) loaded.")

# 1.3. Gallery
FAISS_INDEX = faiss.read_index(cfg.FAISS_INDEX_PATH)
ID2NAME = load_id2name(cfg.ID2NAME_PATH)
logger.info("Global Gallery (Faiss + ID2Name) loaded.")

# 1.4. Utils
_, RECOG_TRANSFORM = get_transforms()
NAME2COLOR = {}
rng = np.random.default_rng(42)
for name in set(ID2NAME.values()):
    color_rgb = tuple(int(c) for c in rng.integers(0, 255, size=3))
    NAME2COLOR[name] = (color_rgb[2], color_rgb[1], color_rgb[0]) # BGR
NAME2COLOR["Unknown"] = (0, 0, 255)
NAME2COLOR["Processing..."] = (192, 192, 192)
NAME2COLOR["Error"] = (0, 255, 255)

# ==================================================
# 2. HÀNG ĐỢI & WORKERS TOÀN CỤC (GLOBAL PIPELINE)
# ==================================================

# --- Pipeline 1: Recognition Worker (Xử lý iResNet) ---
recognition_queue = queue.Queue()
recognition_cache = {}
cache_lock = threading.Lock()
CACHE_TTL_SECONDS = 5.0  # Xóa cache sau 5 giây
last_cache_clean_time = time.time()

def _run_batch_recognition(frame_rgb_copy, faces_to_process):
    """
    Helper: Hàm này sao chép TỪ `realtime_check.py`.
    Nó chạy trong Recognition Worker Thread.
    """
    current_time = time.time()
    batch_tensors = []
    batch_tracker_ids = []
    
    for tracker_id, landmark, xyxy in faces_to_process:
        try:
            aligned_face = align_face(frame_rgb_copy, landmark, xyxy)
            face_tensor = RECOG_TRANSFORM(Image.fromarray(aligned_face))
            batch_tensors.append(face_tensor)
            batch_tracker_ids.append(tracker_id)
        except Exception:
            with cache_lock:
                recognition_cache[tracker_id] = ("Error", 0.0, current_time)

    if not batch_tensors:
        return 

    try:
        batch_tensors = torch.stack(batch_tensors).to(DEVICE)
        with torch.no_grad():
            feats_batch = RECOGNIZER_MODEL(batch_tensors).cpu().numpy()
        
        faiss.normalize_L2(feats_batch)
        scores_batch, indices_batch = FAISS_INDEX.search(feats_batch, 1)

        for i, tracker_id in enumerate(batch_tracker_ids):
            score = scores_batch[i][0]
            idx = indices_batch[i][0]
            name = "Unknown"
            if score > cfg.RECOGNITION_THRESHOLD:
                name = ID2NAME.get(idx, "Unknown")
            
            # CẬP NHẬT CACHE TOÀN CỤC (VỚI KHÓA)
            with cache_lock:
                recognition_cache[tracker_id] = (name, score, current_time)
            
            # Nếu nhận dạng thành công -> Gửi tới DB Worker
            if name != "Unknown":
                db_queue.put({
                    "stream_id": "N/A", # Sẽ được cập nhật sau
                    "tracker_id": tracker_id,
                    "name": name,
                    "score": float(score)
                })

    except Exception as e:
        logger.error(f"[RECOG-WORKER] Lỗi batch recognition: {e}")
        for tracker_id in batch_tracker_ids:
            with cache_lock:
                if tracker_id not in recognition_cache:
                    recognition_cache[tracker_id] = ("Error", 0.0, current_time)

def recognition_worker():
    """Luồng worker chuyên dụng chỉ để chạy iResNet (tránh xung đột GPU)."""
    logger.info("[RECOG-WORKER] Recognition worker started.")
    while True:
        try:
            item = recognition_queue.get()
            if item is None:
                break
            frame_copy, faces_to_process, stream_id = item
            # Gán stream_id cho tất cả các bản ghi DB từ batch này
            _run_batch_recognition(frame_copy, faces_to_process)
            for face_info in faces_to_process:
                tracker_id = face_info[0]
                # Lấy kết quả vừa nhận dạng và gửi vào DB queue
                with cache_lock:
                    result = recognition_cache.get(tracker_id)
                if result and result[0] not in ["Unknown", "Processing...", "Error"]:
                    db_queue.put({
                        "stream_id": stream_id,
                        "tracker_id": tracker_id,
                        "name": result[0],
                        "score": float(result[1])
                    })

            recognition_queue.task_done()
        except Exception as e:
            logger.error(f"[RECOG-WORKER] Lỗi nghiêm trọng: {e}")

# --- Pipeline 2: Database Worker (Ghi SQLite) ---
db_queue = queue.Queue()

def db_worker():
    """Luồng worker chuyên dụng chỉ để ghi DB (SỬ DỤNG MYSQL POOL)."""
    logger.info("[DB-WORKER] MySQL worker started.")
    while True:
        conn = None
        cursor = None
        try:
            item = db_queue.get()
            if item is None:
                break
            
            # Lấy 1 kết nối từ pool (cực nhanh)
            conn = db_pool.get_connection()
            cursor = conn.cursor()
            
            # Câu lệnh INSERT cho MySQL dùng %s
            query = ("INSERT INTO recognition_events "
                     "(stream_id, tracker_id, name, score) "
                     "VALUES (%s, %s, %s, %s)")
            data = (item['stream_id'], item['tracker_id'], item['name'], item['score'])
            
            cursor.execute(query, data)
            conn.commit() # Quan trọng!
            
            db_queue.task_done()
            
        except mysql.connector.Error as err:
            logger.error(f"[DB-WORKER] Lỗi ghi MySQL: {err}")
            # Nếu lỗi (ví dụ: mất kết nối), chúng ta không task_done()
            # và có thể xem xét việc put item trở lại queue
        except Exception as e:
            logger.error(f"[DB-WORKER] Lỗi worker không xác định: {e}")
        finally:
            # Đảm bảo trả kết nối về pool dù thành công hay thất bại
            if cursor:
                cursor.close()
            if conn and conn.is_connected():
                conn.close()

# ===================================================================
# 3. CLASS XỬ LÝ CHO TỪNG LUỒNG (STREAM PROCESSOR)
# ===================================================================

# Class buffer này giúp chỉ lấy frame MỚI NHẤT, bỏ qua frame cũ
class LatestFrameBuffer:
    def __init__(self):
        self._frame = None
        self._lock = threading.Lock()
    def put(self, frame):
        with self._lock: self._frame = frame
    def get(self):
        with self._lock: return None if self._frame is None else self._frame.copy()

class StreamProcessor:
    """Quản lý trạng thái cho MỘT luồng video (camera)."""
    def __init__(self, stream_source, stream_id):
        self.stream_source = stream_source
        self.stream_id = stream_id
        
        self.tracker = ByteTrack(frame_rate=20, lost_track_buffer=30)
        self.frame_buffer = LatestFrameBuffer()
        self.stop_event = threading.Event()
        self.reader_thread = None
        self.web_status = False # Chỉ đọc frame khi có client kết nối
        self.total_count_this_session = 0 # Đếm tạm thời

    def _frame_reader_task(self):
        """Luồng I/O chuyên dụng: Chỉ đọc frame và đưa vào buffer."""
        logger.info(f"[READER-{self.stream_id}] Frame reader task started.")
        backoff = 2
        while not self.stop_event.is_set():
            if not self.web_status:
                time.sleep(1) # Ngủ nếu không có ai xem
                continue

            cap = cv2.VideoCapture(self.stream_source)
            if not cap.isOpened():
                logger.warning(f"[READER-{self.stream_id}] Không thể mở stream. Thử lại sau {backoff}s...")
                time.sleep(backoff)
                backoff = min(backoff * 2, 30)
                continue

            logger.info(f"[READER-{self.stream_id}] Stream opened successfully.")
            backoff = 2
            while self.web_status and not self.stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    logger.warning(f"[READER-{self.stream_id}] Mất kết nối. Đang kết nối lại...")
                    break
                
                # Resize ngay tại luồng đọc để giảm tải
                TARGET_WIDTH = 720
                h, w, _ = frame.shape
                scale = TARGET_WIDTH / w
                new_h, new_w = int(h * scale), TARGET_WIDTH
                frame_small = cv2.resize(frame, (new_w, new_h))
                
                self.frame_buffer.put(frame_small)
                cv2.waitKey(1) # Rất quan trọng cho một số stream RTSP
            cap.release()

    def start_reader(self):
        if self.reader_thread is None or not self.reader_thread.is_alive():
            self.stop_event.clear()
            self.reader_thread = threading.Thread(target=self._frame_reader_task, daemon=True)
            self.reader_thread.start()

    def stop_reader(self):
        self.stop_event.set()
        if self.reader_thread and self.reader_thread.is_alive():
            self.reader_thread.join(timeout=2)
        self.web_status = False

# ===================================================================
# 4. LOGIC XỬ LÝ CHÍNH (CHẠY TRONG AI THREAD POOL)
# ===================================================================

def _get_detections_for_tracker(frame_rgb):
    """Helper: Sao chép từ realtime_check.py, nhưng dùng model TOÀN CỤC."""
    results = DETECTOR_MODEL(frame_rgb, device=DEVICE, verbose=False)
    # ... (Toàn bộ logic xử lý results giữ nguyên y hệt file gốc của bạn) ...
    boxes_xyxy = []
    confs = []
    landmarks = []
    for res in results:
        if res.boxes is None: continue
        xyxy_cpu = res.boxes.xyxy.cpu().numpy()
        confs_cpu = res.boxes.conf.cpu().numpy()
        if res.keypoints is not None and len(res.keypoints.xy) > 0:
            keypoints_xy = res.keypoints.xy.cpu().numpy()
        else: 
            keypoints_xy = np.zeros((len(xyxy_cpu), 5, 2))
        boxes_xyxy.append(xyxy_cpu)
        confs.append(confs_cpu)
        landmarks.append(keypoints_xy)
    if not boxes_xyxy: return None
    all_boxes_xyxy = np.concatenate(boxes_xyxy)
    all_confs = np.concatenate(confs)
    all_landmarks = np.concatenate(landmarks)
    detections = Detections(xyxy=all_boxes_xyxy, confidence=all_confs)
    detections.data['landmarks'] = all_landmarks
    return detections

def _clean_global_cache(current_time):
    """Dọn dẹp cache TOÀN CỤC."""
    global last_cache_clean_time
    if current_time - last_cache_clean_time <= 30.0:
        return # Chỉ dọn 30s một lần
        
    ids_to_remove = []
    with cache_lock:
        try:
            for tracker_id, (_, _, timestamp) in recognition_cache.items():
                if current_time - timestamp > CACHE_TTL_SECONDS:
                    ids_to_remove.append(tracker_id)
            for tracker_id in ids_to_remove:
                if tracker_id in recognition_cache:
                    del recognition_cache[tracker_id]
        except RuntimeError:
            pass # Bỏ qua nếu dict thay đổi
    last_cache_clean_time = current_time
    if ids_to_remove:
        logger.info(f"[CACHE] Dọn dẹp {len(ids_to_remove)} ID quá hạn.")

def process_and_draw_frame(processor: StreamProcessor, frame: np.ndarray):
    """
    Hàm này chạy trong AI Thread Pool (từ run_in_threadpool).
    Nó thực hiện Detect + Track và gửi yêu cầu Recognition.
    """
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    current_time = time.time()
    
    # 1. Dọn cache (chỉ chạy định kỳ)
    _clean_global_cache(current_time)

    # 2. Detect và Track (Nhanh)
    detections = _get_detections_for_tracker(frame_rgb)
    if detections is None:
        return frame, processor.total_count_this_session

    tracked_detections = processor.tracker.update_with_detections(detections)

    faces_to_process_queue = [] # Batch cho recognition worker

    # 3. Phân loại mặt
    new_faces_found = False
    for i in range(len(tracked_detections)):
        xyxy = tracked_detections.xyxy[i]
        tracker_id = tracked_detections.tracker_id[i]
        landmark = tracked_detections.data['landmarks'][i]
        
        with cache_lock:
            cached_result = recognition_cache.get(tracker_id)

        if cached_result is None:
            # Khuôn mặt HOÀN TOÀN MỚI
            faces_to_process_queue.append((tracker_id, landmark, xyxy))
            # Đánh dấu tạm thời để tránh gửi liên tục
            with cache_lock:
                recognition_cache[tracker_id] = ("Processing...", 0.0, current_time)
            new_faces_found = True
        else:
            # Đã biết hoặc đang xử lý. Cập nhật timestamp để không bị xóa
            name, score, _ = cached_result
            if name == "Unknown": # Nếu trước đó là Unknown, thử nhận dạng lại
                new_faces_found = True
                faces_to_process_queue.append((tracker_id, landmark, xyxy))
            
            with cache_lock:
                recognition_cache[tracker_id] = (name, score, current_time)
            
            # Đếm nếu là người đã xác định
            if name not in ["Processing...", "Unknown", "Error"]:
                processor.total_count_this_session += 1 # Đây chỉ là logic đếm ví dụ

    # 4. GỬI YÊU CẦU NHẬN DẠNG (NON-BLOCKING)
    if new_faces_found and faces_to_process_queue:
        # Gửi toàn bộ batch + frame gốc sang Recognition Worker
        recognition_queue.put((frame_rgb.copy(), faces_to_process_queue, processor.stream_id))

    # 5. Vẽ lên frame (Luôn đọc từ cache)
    annotated_frame = frame.copy()
    for detection in tracked_detections:
        xyxy = detection[0]
        tracker_id = detection[4]
        
        # Đọc cache MỘT LẦN KHI VẼ (có khóa)
        with cache_lock:
            name, score, _ = recognition_cache.get(tracker_id, ("Processing...", 0.0, 0))

        label = f"{name} ({score:.2f})"
        x1, y1, x2, y2 = [int(coord) for coord in xyxy]
        color_bgr = NAME2COLOR.get(name, (0, 0, 255)) 

        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color_bgr, 2)
        text_y = y1 - 10 if y1 - 10 > 10 else y2 + 20
        cv2.putText(annotated_frame, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2)

    return annotated_frame, processor.total_count_this_session


# =====================================
# 5. FASTAPI APPLICATION VÀ WEBSOCKETS
# =====================================
app = FastAPI()
# Tạo AI Thread Pool
AI_EXECUTOR = ThreadPoolExecutor(max_workers=os.cpu_count() or 4) 

# --- Cấu hình các luồng đầu vào ---
STREAM_SOURCES = {
    0: "rtsp://rtsp-server:8554/mystream1",
    1: "rtsp://rtsp-server:8554/mystream2",
}

# Khởi tạo các processor cho mỗi nguồn
PROCESSORS = {
    stream_id: StreamProcessor(source, stream_id) 
    for stream_id, source in STREAM_SOURCES.items()
}

@app.on_event("startup")
async def startup_event():
    logger.info("Initializing database...")
    initialize_database()
    
    """Khởi động tất cả các luồng worker nền."""
    # 1. Khởi động DB Worker
    threading.Thread(target=db_worker, daemon=True).start()
    # 2. Khởi động Recognition Worker
    threading.Thread(target=recognition_worker, daemon=True).start()
    # 3. Khởi động tất cả các Frame Reader
    for proc in PROCESSORS.values():
        proc.start_reader()
    logger.info("All background workers and stream readers started.")

@app.on_event("shutdown")
async def shutdown_event():
    """Dọn dẹp tài nguyên khi tắt server."""
    logger.info("Shutting down...")
    for proc in PROCESSORS.values():
        proc.stop_reader()
    recognition_queue.put(None) # Gửi tín hiệu dừng
    db_queue.put(None)          # Gửi tín hiệu dừng
    AI_EXECUTOR.shutdown(wait=True)
    logger.info("Shutdown complete.")

# Phục vụ file HTML
@app.get("/")
async def get_index():
    # Đọc file HTML từ ổ đĩa
    html_path = os.path.join(current_dir, "index.html")
    if not os.path.exists(html_path):
        return HTMLResponse("<html><body><h1>Lỗi</h1><p>Không tìm thấy file index.html</p></body></html>", status_code=500)
    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    
    # Tự động tạo các khối video trong HTML dựa trên cấu hình STREAM_SOURCES
    stream_blocks = ""
    for stream_id in STREAM_SOURCES.keys():
        stream_blocks += f"""
        <div class="stream-wrapper">
            <h3>Camera {stream_id}</h3>
            <canvas id="video-canvas-{stream_id}" class="video-canvas"></canvas>
            <div id="data-display-{stream_id}" class="data-display">Count: --</div>
        </div>
        """
    return HTMLResponse(html_content.replace("", stream_blocks))

# Phục vụ file JavaScript
@app.get("/static/main.js")
async def get_js():
    js_path = os.path.join(current_dir, "static", "main.js")
    if not os.path.exists(js_path):
        return HTMLResponse("alert('main.js not found!');", media_type="application/javascript", status_code=404)
    with open(js_path, "r", encoding="utf-8") as f:
        js_content = f.read()
    
    # Tự động chèn các ID stream vào JavaScript
    stream_ids_js_array = json.dumps(list(STREAM_SOURCES.keys()))
    js_content = js_content.replace("['STREAM_IDS_PLACEHOLDER']", stream_ids_js_array)
    return HTMLResponse(js_content, media_type="application/javascript")


@app.websocket("/ws/{stream_id}")
async def websocket_endpoint(websocket: WebSocket, stream_id: int):
    """Điểm cuối WebSocket: Nhận frame từ Reader, đẩy vào AI Pool, stream kết quả."""
    if stream_id not in PROCESSORS:
        await websocket.close(code=1008, reason="Invalid Stream ID")
        return

    await websocket.accept()
    processor = PROCESSORS[stream_id]
    processor.web_status = True # Báo cho Reader Thread bắt đầu đọc
    logger.info(f"[WS-{stream_id}] Client connected. Reader activated.")

    try:
        while websocket.client_state == WebSocketState.CONNECTED:
            # 1. Lấy frame mới nhất từ Reader Thread Buffer (Non-blocking)
            frame = processor.frame_buffer.get() 
            
            if frame is None:
                await asyncio.sleep(0.01) # Chờ frame nếu buffer rỗng
                continue

            # 2. Đẩy tác vụ AI (nặng) vào AI Thread Pool (Non-blocking)
            processed_frame, count = await run_in_threadpool(
                process_and_draw_frame, processor, frame
            )

            # 3. Mã hóa và Gửi kết quả về trình duyệt (Async)
            ok, buf = cv2.imencode('.jpg', processed_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            if not ok:
                continue

            base64_image = base64.b64encode(buf).decode('utf-8')
            
            # Gửi gói JSON chứa ảnh và dữ liệu
            message = {
                'image': base64_image,
                'count': count, # Gửi dữ liệu đếm
                'timestamp': datetime.now().isoformat()
            }
            await websocket.send_json(message)
            
            # Giải phóng event loop một chút
            await asyncio.sleep(0.001) 

    except WebSocketDisconnect:
        logger.info(f"[WS-{stream_id}] Client disconnected.")
    except Exception as e:
        logger.error(f"[WS-{stream_id}] Error: {e}")
    finally:
        processor.web_status = False # Báo Reader Thread dừng lại
        logger.info(f"[WS-{stream_id}] Connection closed. Reader deactivated.")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)
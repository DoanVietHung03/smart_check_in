# main.py (ĐÃ TÍCH HỢP LOGIC NHẬN DẠNG KHUÔN MẶT)
import warnings
warnings.filterwarnings('ignore')

import os
import sys
import cv2
import time
import math
import torch
import faiss
import numpy as np
from PIL import Image
import asyncio
import threading
import queue
import uvicorn
import base64
import json
from collections import deque, Counter 
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

# FastAPI và các thành phần liên quan
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.concurrency import run_in_threadpool
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketState
from contextlib import asynccontextmanager

# Các thư viện xử lý ảnh và AI
from ultralytics import YOLO
from supervision import ByteTrack, Detections

# Hàm khởi tạo DB
from .db_manager import initialize_database, db_insert

import logging
# ====================== LOGGING ======================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('FaceRec-Server')

# === THÊM MODULES CHO FACE REC (YÊU CẦU PYTHONPATH=/app TỪ DOCKER) ===
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  
setting_and_train_dir = os.path.join(project_root, "Setting_and_Train")
sys.path.append(setting_and_train_dir)

from config import cfg
from model import iresnet34
from utils import align_face, get_transforms, load_id2name, FaceDetector_YOLO, FaceDetector_RetinaFace

logger.info("Imports completed. Loading models...")

# ====================== CẤU HÌNH (CONFIG) ======================
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'

RTSP_URLS = [
    'rtsp://rtsp-server:8554/mystream1',
    'rtsp://rtsp-server:8554/mystream2'
]

# IO / Model
RESIZE_PERCENT = 70
LOST_TRACK_BUFFER = 50
USE_LATEST_ONLY = True
JPEG_QUALITY = 80
CACHE_TTL_SECONDS = 5.0 # Thời gian xóa cache
SESSION_TIMEOUT_SECONDS = 300.0 # (5 phút) Xóa người khỏi sidebar nếu không thấy

# ===================== TẢI MODEL TOÀN CỤC =========================
DEVICE = torch.device(cfg.DEVICE)

# 1. Detector
if cfg.DETECTOR_TYPE == 'yolo':
    DETECTOR_MODEL = FaceDetector_YOLO()
    logger.info("Global Face Detector (YOLO) loaded.")
elif cfg.DETECTOR_TYPE == 'retinaface':
    DETECTOR_MODEL = FaceDetector_RetinaFace()
    logger.info("Global Face Detector (RetinaFace) loaded.")
else:
    raise ValueError("Invalid DETECTOR_TYPE in config.py. Choose 'yolo' or 'retinaface'.") 

# 2. Recognizer
RECOGNIZER_MODEL = iresnet34(fp16=False).to(DEVICE)
ckpt = torch.load(cfg.PRETRAINED_RECOGNITION_MODEL_PATH, map_location=DEVICE)
RECOGNIZER_MODEL.load_state_dict(ckpt, strict=True)
RECOGNIZER_MODEL.eval()
logger.info(f"Global Face Recognizer (iResNet) loaded from: {cfg.PRETRAINED_RECOGNITION_MODEL_PATH}")

# 3. Gallery
FAISS_INDEX = faiss.read_index(cfg.FAISS_INDEX_PATH)
ID2NAME = load_id2name(cfg.ID2NAME_PATH)
_, RECOG_TRANSFORM = get_transforms()
logger.info("Global Gallery (Faiss + ID2Name) loaded.")

# 4. Bảng màu
NAME2COLOR = {}
rng = np.random.default_rng(42)
for name in set(ID2NAME.values()):
    color_rgb = tuple(int(c) for c in rng.integers(0, 255, size=3))
    NAME2COLOR[name] = (color_rgb[2], color_rgb[1], color_rgb[0]) # BGR
NAME2COLOR.update({"Unknown": (0, 0, 255), "Processing...": (192, 192, 192), "Error": (0, 255, 255)})

NAME_TO_WEB_PATH = {}
try:
    with open(cfg.NAME2PATH_PATH, 'r', encoding='utf-8') as f:
        name2path_fs = json.load(f) # Đường dẫn hệ thống tập tin
    
    for name, fs_path in name2path_fs.items():
        # Chuyển đổi sang định dạng URL và thêm tiền tố mount
        web_path = f"/gallery_images/{fs_path}"
        NAME_TO_WEB_PATH[name] = web_path
    
    logger.info(f"Đã tải {len(NAME_TO_WEB_PATH)} đường dẫn ảnh gallery và chuyển đổi sang URL web.")

except Exception as e:
    logger.error(f"Không thể tải hoặc xử lý name2path.json: {e}. Sidebar thumbnail sẽ không hoạt động.")

# ===================== KHỞI TẠO DB =========================
stream_ids = [i for i, _ in enumerate(RTSP_URLS)]
try:
    initialize_database(stream_ids)
except Exception as e:
    logger.error(f"Không thể khởi tạo CSDL: {e}")
    exit()

# ====================== WORKERS NỀN (PIPELINE) ======================
# CHÚNG TA SẼ CÓ 2 WORKER: 1 CHO RECOGNITION, 1 CHO DATABASE

# --- Pipeline 1: Database Worker ---
db_queue = queue.Queue()

def db_worker():
    while True:
        item = db_queue.get()
        if item is None: break
        stream_id, detection_event = item
        try:
            db_insert(detection_event, stream_id)
        except Exception as e:
            logger.error(f"Lỗi khi insert DB cho stream {stream_id}: {e}")
        finally:
            db_queue.task_done()

# --- Pipeline 2: Recognition Worker (LOGIC VOTING NÂNG CẤP) ---
recognition_queue = queue.Queue()
recognition_cache = {} # Cache nhận dạng (ID -> Tên)
cache_lock = threading.Lock()
last_cache_clean_time = time.time()

def recognition_worker_thread():
    """Luồng chuyên dụng chỉ để chạy iResNet, tránh xung đột GPU."""
    logger.info("[RECOG-WORKER] Recognition worker started.")
    while True:
        try:
            item = recognition_queue.get()
            if item is None: break
            
            frame_copy, faces_to_process, stream_id = item
            current_time = time.time()
            batch_tensors = []
            batch_keys = []

            for composite_key, landmark, xyxy in faces_to_process:
                try:
                    aligned_face = align_face(frame_copy, landmark, xyxy)
                    face_tensor = RECOG_TRANSFORM(Image.fromarray(aligned_face))
                    batch_tensors.append(face_tensor)
                    batch_keys.append(composite_key)
                except Exception:
                    with cache_lock:
                        recognition_cache[composite_key] = ("Error", 0.0, current_time)

            if not batch_tensors:
                recognition_queue.task_done()
                continue
            
            # Chạy Batch Inference
            batch_tensors = torch.stack(batch_tensors).to(DEVICE)
            with torch.no_grad():
                feats_batch = RECOGNIZER_MODEL(batch_tensors).cpu().numpy()
            
            faiss.normalize_L2(feats_batch)
            scores_batch, indices_batch = FAISS_INDEX.search(feats_batch, 1)

            for i, composite_key in enumerate(batch_keys):
                score = scores_batch[i][0]
                idx = indices_batch[i][0]
                name = "Unknown"
                if score > cfg.RECOGNITION_THRESHOLD:
                    name = ID2NAME.get(idx, "Unknown")
                
                # Cập nhật Cache toàn cục
                with cache_lock:
                    recognition_cache[composite_key] = (name, score, current_time) # Dùng key tổng hợp
                
                # Nếu nhận dạng thành công -> Gửi tới DB Worker
                if name != "Unknown":
                    db_event = {
                        "timestamp": datetime.now(), # Lấy thời gian hiện tại
                        "stream_id": stream_id,
                        "tracker_id": composite_key[1],
                        "name": name,
                        "score": float(score)
                    }
                    db_queue.put((stream_id, db_event)) # Gửi (stream_id, event) cho db_worker

            recognition_queue.task_done()
        except Exception as e:
            logger.error(f"[RECOG-WORKER] Lỗi nghiêm trọng: {e}")
            recognition_queue.task_done()

# ====================== VIDEO PROCESSOR CLASS ======================
class VideoProcessor:
    def __init__(self, stream_source, stream_id, shared_detector):
        self.stream_source = stream_source
        self.stream_id = stream_id
        self.detector = shared_detector # Dùng chung detector
        
        self.fps = self._probe_fps()
        logger.info(f'Stream {stream_id} probed FPS: {self.fps}')
        self.tracker = ByteTrack(frame_rate=int(self.fps), lost_track_buffer=LOST_TRACK_BUFFER)
        
        # State
        self.frame_idx = 0
        self.frame_skip = 1 # Bắt đầu bằng việc xử lý mọi frame

        # Buffer & Thread (Từ template)
        self.frame_buffer = self.LatestFrameBuffer() if USE_LATEST_ONLY else deque(maxlen=10)
        self.stop_event = threading.Event()
        self.reader_thread = None
        self.web_status = False
        
        # State mới để lưu giữ những người đã thấy (thread-safe)
        self.recently_seen = {} # Key: tracker_id, Value: {data cho UI}
        self.session_lock = threading.Lock()

    class LatestFrameBuffer:
        def __init__(self):
            self._frame = None
            self._lock = threading.Lock()
        
        def put(self, frame):
            with self._lock:
                self._frame = frame
        
        def get(self):
            with self._lock:
                return None if self._frame is None else self._frame.copy()

    def _probe_fps(self):
        try:
            cap = cv2.VideoCapture(self.stream_source, cv2.CAP_FFMPEG)
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            return fps if fps and not math.isnan(fps) and fps > 0 else 25.0
        except Exception: return 25.0

    def _frame_reader_task(self):
        backoff = 2
        while not self.stop_event.is_set():
            if not self.web_status:
                time.sleep(1)
                continue

            cap = cv2.VideoCapture(self.stream_source, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 5)
            if not cap.isOpened():
                logger.error(f'Không thể mở RTSP stream {self.stream_id}. Thử lại sau {backoff}s...')
                time.sleep(backoff)
                backoff = min(backoff * 2, 30)
                continue
            
            backoff = 2
            logger.info(f"RTSP stream {self.stream_id} đã mở.")
            while self.web_status and not self.stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    logger.error(f'Stream {self.stream_id} bị ngắt. Đang kết nối lại...')
                    break
                if RESIZE_PERCENT != 100:
                    h, w = frame.shape[:2]
                    nh, nw = int(h * RESIZE_PERCENT / 100), int(w * RESIZE_PERCENT / 100)
                    frame = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
                self.frame_buffer.put(frame)
            cap.release()

    def start_reader(self):
        if self.reader_thread is None or not self.reader_thread.is_alive():
            self.stop_event.clear()
            self.reader_thread = threading.Thread(target=self._frame_reader_task, daemon=True)
            self.reader_thread.start()
            logger.info(f"Frame reader thread cho stream {self.stream_id} đã bắt đầu.")

    def stop_reader(self):
        self.stop_event.set()
        if self.reader_thread and self.reader_thread.is_alive():
            self.reader_thread.join(timeout=2)
        logger.info(f"Frame reader cho stream {self.stream_id} đã dừng.")

    def set_web_status(self, status: bool):
        self.web_status = status
        self.frame_idx = 0
        logger.info(f"Trạng thái Web cho stream {self.stream_id} được đặt: {status}")
        if not status:
            self.stop_reader()

    def process_and_draw_frame(self, frame: cv2.Mat):
        """
        Tạo và duy trì một "Nhật ký phiên" (self.recently_seen) của những người được nhận diện,
        chỉ xóa họ nếu họ vắng mặt quá SESSION_TIMEOUT_SECONDS.
        """
        start_time = time.perf_counter()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        current_time = time.time()
        
        # 1. Dọn Cache quá hạn
        global last_cache_clean_time
        if current_time - last_cache_clean_time > 30.0:
            ids_to_remove = []
            with cache_lock:
                try:
                    for key, (_, _, timestamp) in recognition_cache.items(): 
                        if current_time - timestamp > CACHE_TTL_SECONDS: ids_to_remove.append(key)
                    for key in ids_to_remove:
                        if key in recognition_cache: del recognition_cache[key]
                except RuntimeError: pass
            last_cache_clean_time = current_time
            if ids_to_remove: logger.info(f"[CACHE] Dọn dẹp {len(ids_to_remove)} ID quá hạn.")

        # 2. Detect & Track
        if cfg.DETECTOR_TYPE == 'yolo':
            results = self.detector.model(frame_rgb, device=DEVICE, verbose=False, conf=cfg.DETECTION_CONFIDENCE)[0]
            detections_sv = Detections.from_ultralytics(results)
            if results.keypoints is not None and len(results.keypoints.xy) > 0:
                detections_sv.data['landmarks'] = results.keypoints.xy.cpu().numpy()
            else: # Nếu không có landmarks, tạo mảng rỗng
                detections_sv.data['landmarks'] = np.zeros((len(detections_sv), 5, 2))
        else: # retinaface
            boxes, landmarks = self.detector.detect(frame_rgb)
            valid_detections_data = [
                (box, landmark) for box, landmark in zip(boxes, landmarks) if box
            ]

            if valid_detections_data:
                # Unzip the filtered data back into separate lists
                valid_boxes, valid_landmarks = zip(*valid_detections_data)
                
                detections_sv = Detections(
                    xyxy=np.array(valid_boxes, dtype=float),
                    confidence=np.ones(len(valid_boxes)),
                    data={'landmarks': np.array(valid_landmarks)}
                )
            else:
                detections_sv = Detections.empty()
                
        tracked_detections = self.tracker.update_with_detections(detections_sv)

        faces_to_process_queue = []
        new_faces_found = False
        
        # Set các tên đã được xử lý trong frame này (để tránh xử lý 1 tên 2 lần)
        names_processed_this_frame = set() 

        # 3. Vòng lặp chính: Cập nhật cache, Gửi đi nhận dạng, VÀ CẬP NHẬT PHIÊN SIDEBAR
        for i in range(len(tracked_detections)):
            xyxy = tracked_detections.xyxy[i]
            tracker_id = int(tracked_detections.tracker_id[i])
            landmark = tracked_detections.data['landmarks'][i]

            # Tạo key tổng hợp (stream_id, tracker_id)
            composite_key = (self.stream_id, tracker_id)
            # Đọc cache bằng key tổng hợp
            with cache_lock: cached_result = recognition_cache.get(composite_key)

            if cached_result is None:
                # Gửi key tổng hợp đi nhận dạng
                faces_to_process_queue.append((composite_key, landmark, xyxy))
                with cache_lock: recognition_cache[composite_key] = ("Processing...", 0.0, current_time)
                new_faces_found = True
            else:
                name, score, _ = cached_result
                if name == "Unknown":
                    new_faces_found = True
                    faces_to_process_queue.append((composite_key, landmark, xyxy))
                
                with cache_lock: # Cập nhật timestamp cho cache
                    recognition_cache[composite_key] = (name, score, current_time)

                if name not in ["Processing...", "Unknown", "Error"] and name not in names_processed_this_frame:
                    with self.session_lock:
                        if name not in self.recently_seen: # Key bằng TÊN
                            # LẦN ĐẦU THẤY TÊN NÀY:
                            logger.info(f"[Stream {self.stream_id}] Phát hiện người mới vào phiên: {name} (ID: {tracker_id})")
                            
                            color_bgr = NAME2COLOR.get(name, (0,0,255))
                            color_css = f"rgb({color_bgr[2]}, {color_bgr[1]}, {color_bgr[0]})"
                            thumb_web_path = NAME_TO_WEB_PATH.get(name, None)
                            
                            self.recently_seen[name] = { # Key bằng TÊN
                                "id": name, # ID cho Javascript giờ là TÊN
                                "name": name,
                                "color": color_css,
                                "thumb": thumb_web_path,
                                "last_seen": current_time
                            }
                        else:
                            # ĐÃ THẤY TÊN NÀY: Chỉ cập nhật timestamp
                            self.recently_seen[name]['last_seen'] = current_time
                    
                    names_processed_this_frame.add(name) # Đánh dấu đã xử lý tên này

        if new_faces_found and faces_to_process_queue:
            recognition_queue.put((frame_rgb.copy(), faces_to_process_queue, self.stream_id))

        # 4. Vẽ lên frame
        annotated_frame = frame.copy()
        for detection in tracked_detections:
            xyxy, _, _, _, tracker_id_raw, _ = detection
            tracker_id = int(tracker_id_raw) # Ép kiểu int
            
            # Đọc cache để vẽ
            composite_key = (self.stream_id, tracker_id)
            with cache_lock: name, score, _ = recognition_cache.get(composite_key, ("Processing...", 0.0, 0))
            
            label = f"{name} ({score:.2f})"
            x1, y1, x2, y2 = [int(coord) for coord in xyxy]
            color_bgr = NAME2COLOR.get(name, (0, 0, 255))
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color_bgr, 2)
            text_y = y1 - 10 if y1 - 10 > 10 else y2 + 20
            cv2.putText(annotated_frame, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2)

        # 5. DỌN DẸP PHIÊN SIDEBAR (Xóa người cũ)
        names_to_remove_from_session = [] # Giờ là danh sách TÊN
        with self.session_lock:
            # === SỬA LỖI B ===
            for name, data in self.recently_seen.items(): # Key là TÊN
                if current_time - data['last_seen'] > SESSION_TIMEOUT_SECONDS:
                    names_to_remove_from_session.append(name)
            
            for name in names_to_remove_from_session:
                del self.recently_seen[name]
                logger.info(f"[Stream {self.stream_id}] Xóa {name} khỏi phiên (timeout).")
            # === KẾT THÚC SỬA LỖI B ===

        # 6. TẠO DỮ LIỆU GỬI ĐI
        identities_payload = list(self.recently_seen.values())
        processing_time = time.perf_counter() - start_time
        
        return annotated_frame, processing_time, identities_payload

    def tune_frame_skip_by_time(self, processing_time: float):
        frame_interval = 1.0 / self.fps
        if processing_time > frame_interval * 1.5: # LOAD_THRESHOLD_HIGH
            self.frame_skip = min(self.frame_skip + 1, 8) # FRAME_SKIP_MAX
        elif processing_time < frame_interval * 0.5: # LOAD_THRESHOLD_LOW
            self.frame_skip = max(self.frame_skip - 1, 1) # FRAME_SKIP_MIN

# ====================== FASTAPI APP ======================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- STARTUP LOGIC ---
    logger.info("Initializing database...")
    if not initialize_database(stream_ids):
         logger.critical("FATAL: Database initialization failed.")
         # Trong production, bạn có thể muốn raise Exception ở đây
    else:
        logger.info("Database schema initialized successfully.")

    # Chỉ khởi động worker 1 lần duy nhất ở đây
    threading.Thread(target=db_worker, daemon=True).start()
    threading.Thread(target=recognition_worker_thread, daemon=True).start()
    
    for proc in PROCESSORS.values():
        proc.start_reader()
    logger.info("All background workers and stream readers started.")
    
    yield # Ứng dụng chạy ở đây

    # --- SHUTDOWN LOGIC ---
    logger.info("Shutting down...")
    for proc in PROCESSORS.values():
        proc.stop_reader()
    recognition_queue.put(None); db_queue.put(None)
    AI_EXECUTOR.shutdown(wait=True)
    logger.info("Shutdown complete.")

app = FastAPI(lifespan=lifespan)
AI_EXECUTOR = ThreadPoolExecutor(max_workers=max(4, min(os.cpu_count() or 4, len(RTSP_URLS) * 2))) # Giới hạn số luồng AI

PROCESSORS = {
    i: VideoProcessor(source, i, DETECTOR_MODEL)
    for i, source in enumerate(RTSP_URLS)
}

# Đăng ký StaticFiles
current_dir = os.path.dirname(os.path.abspath(__file__))
static_dir_path = os.path.join(current_dir, "static")
app.mount("/static", StaticFiles(directory=static_dir_path), name="static")

try:
    app.mount("/gallery_images", StaticFiles(directory=cfg.DATA_ROOT), name="gallery_images")
    logger.info(f"Serving static gallery images from: {cfg.DATA_ROOT}")
except Exception as e:
    logger.error(f"Failed to mount gallery directory: {e}. Thumbnails will NOT load.")

# HTML giao diện
HTML = """<!doctype html>
<html>
<head>
    <meta charset='utf-8'>
    <title>Face Recognition Streams</title>
    <style>
        body { background: #111; color: #eee; font-family: sans-serif; margin: 0; padding: 0; }
        h2 { text-align: center; color: #00FF00; margin-top: 15px; }
        
        /* Layout chính 2 cột */
        .main-container {
            display: flex;
            flex-direction: row;
            width: 100%%;
            height: calc(100vh - 50px); /* Chiều cao tối đa trừ đi tiêu đề */
        }
        
        /* CỘT BÊN TRÁI: Video Streams */
        .video-panel {
            flex: 3; /* Chiếm 3/4 không gian */
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            align-content: flex-start;
            overflow-y: auto;
            background: #0a0a0a;
            padding: 10px;
        }
        .stream-wrapper {
            text-align: center;
            margin: 10px;
            padding: 5px;
            border: 1px solid #333;
            border-radius: 5px;
            background: #1a1a1a;
        }
        .stream-wrapper h3 { margin: 5px 0; }
        canvas { display: block; width: auto; max-width: 100%%; height: auto; image-rendering: pixelated; border-radius: 4px; } 

        /* CỘT BÊN PHẢI: Sidebar Nhận diện */
        .sidebar-panel {
            flex: 1; /* Chiếm 1/4 không gian */
            display: flex;
            flex-direction: column;
            overflow-y: hidden;
            background: #1E1E1E;
            border-left: 2px solid #444;
        }
        .sidebar-wrapper {
            overflow-y: auto; /* Cho phép cuộn nếu danh sách quá dài */
            flex-grow: 1;
        }
        .sidebar-stream-group {
            border-bottom: 1px dashed #555;
            padding-bottom: 10px;
            margin-bottom: 10px;
        }
        .sidebar-stream-group h3 { text-align: center; color: #ddd; }
        .sidebar-content {
            display: flex;
            flex-wrap: wrap; /* Tự động xuống hàng các thumbnail */
            justify-content: center;
        }
        .sidebar-empty { padding: 20px; text-align: center; color: #888; }
        
        /* Thẻ thumbnail */
        .identity-card {
            width: 100px;
            margin: 5px;
            text-align: center;
            background: #2a2a2a;
            border-radius: 4px;
            overflow: hidden;
        }
        .identity-thumb {
            width: 100%%; /* Cố định chiều rộng */
            height: 100px; /* Cố định chiều cao */
            object-fit: cover; /* Đảm bảo ảnh lấp đầy, không bị méo */
        }
        .identity-name {
            font-weight: bold;
            font-size: 0.9em;
            margin: 5px 0;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
    </style>
</head>
<body>
    <h2>Multi-Stream Face Recognition Server</h2>
    <div class='main-container'>
        <div class='video-panel video-container' data-stream-count='%d'>
            %s
        </div>
        <div class='sidebar-panel'>
            <div class="sidebar-wrapper"> %s
            </div>
        </div>
    </div>
    <script src='/static/main.js'></script>
</body>
</html>
"""

# Tạo 2 khối HTML: Một cho Video, một cho Sidebar
stream_html_videos = ""
stream_html_sidebars = ""

for i in range(len(RTSP_URLS)):
    # Tạo khối video cho cột trái
    stream_html_videos += f"""
    <div class="stream-wrapper">
        <h3>Camera {i}</h3>
        <canvas id='video-stream-{i}' class='video'></canvas>
    </div>
    """
    # Tạo khối chứa (rỗng) cho sidebar cột phải
    stream_html_sidebars += f"""
    <div class="sidebar-stream-group">
        <h3>Camera {i} - Nhận diện</h3>
        <div class="sidebar-content" id="identity-sidebar-{i}">
             <p class="sidebar-empty">Đang chờ kết nối...</p>
        </div>
    </div>
    """

# Điền 3 giá trị vào HTML chính
HTML_RESPONSE = HTML % (len(RTSP_URLS), stream_html_videos, stream_html_sidebars)

@app.get('/')
async def index():
    return HTMLResponse(HTML_RESPONSE)

@app.websocket('/ws/{stream_id}')
async def ws_endpoint(ws: WebSocket, stream_id: int):
    if not (0 <= stream_id < len(PROCESSORS)):
        await ws.close(code=1008, reason="Invalid stream ID")
        return

    await ws.accept()
    processor = PROCESSORS[stream_id]
    processor.set_web_status(True)

    try:
        while ws.client_state == WebSocketState.CONNECTED:
            frame = processor.frame_buffer.get()
            if frame is None:
                await asyncio.sleep(0.01)
                continue

            processor.frame_idx += 1
            processed_frame = None
            visible_identities = []

            if processor.frame_idx % processor.frame_skip == 0:
                # Chạy logic Face Rec trong ThreadPool, hàm này giờ trả về 3 giá trị
                processed_frame, processing_time, visible_identities = await run_in_threadpool(
                    processor.process_and_draw_frame, frame
                )
                processor.tune_frame_skip_by_time(processing_time)
            else:
                # Nếu skip, chúng ta chỉ cần vẽ lại frame cũ với box cũ
                # (Chúng ta có thể bỏ qua phần này và chỉ gửi khi xử lý để giảm tải)
                pass 

            # Chỉ gửi frame nếu nó đã được xử lý
            if processed_frame is not None:
                ok, buf = cv2.imencode('.jpg', processed_frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
                if ok:
                    base64_image = base64.b64encode(buf).decode('utf-8')
                    message = {
                        'image': base64_image,
                        'identities': visible_identities,  # Lấy danh sách các khuôn mặt nhận diện
                        'stream_id': processor.stream_id
                    }
                    await ws.send_text(json.dumps(message))
                else:
                    logger.error(f"Không thể encode frame cho stream {processor.stream_id}")

            await asyncio.sleep(0.001)

    except WebSocketDisconnect:
        logger.info(f"Client đã ngắt kết nối khỏi stream {stream_id}.")
    except Exception as e:
        logger.error(f'Lỗi WebSocket trên stream {stream_id}: {e}', exc_info=True)
    finally:
        processor.set_web_status(False)
        logger.info(f"Kết nối WebSocket cho stream {processor.stream_id} đã đóng.")

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000, workers=1)
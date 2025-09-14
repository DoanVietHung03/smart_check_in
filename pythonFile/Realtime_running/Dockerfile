# Dùng base image Python 3.12-slim 
FROM python:3.12-slim

# Cài đặt các thư viện hệ thống bắt buộc cho OpenCV và PyTorch
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Cài đặt PyTorch với hỗ trợ CUDA 12.1 (nếu GPU có sẵn)
RUN pip install --no-cache-dir torch==2.5.1+cu121 torchvision==0.20.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121

# Copy và cài đặt requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Đặt thư mục làm việc thành /app
WORKDIR /app

# Copy TOÀN BỘ dự án (bao gồm pythonFile/, pretrained_model_weights/, v.v.) vào /app
COPY . .

# Thông báo cổng FastAPI sẽ chạy 
EXPOSE 8000

# Lệnh mặc định để chạy server.py (vì server.py tự nó gọi uvicorn.run)
CMD ["python", "server.py"]
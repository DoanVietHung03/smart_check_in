// static/main.js

document.addEventListener("DOMContentLoaded", () => {
    console.log("DOM loaded. Initializing WebSocket connections...");

    // Mảng này sẽ được thay thế bởi server (server.py)
    const STREAM_IDS = ['STREAM_IDS_PLACEHOLDER'];

    if (STREAM_IDS[0] === 'STREAM_IDS_PLACEHOLDER') {
        console.error("Lỗi: Server chưa chèn Stream IDs. Kiểm tra file server.py, hàm get_js.");
        return;
    }

    // Lặp qua từng ID stream và kết nối
    STREAM_IDS.forEach(streamId => {
        const canvas = document.getElementById(`video-canvas-${streamId}`);
        const dataDisplay = document.getElementById(`data-display-${streamId}`);
        
        if (!canvas || !dataDisplay) {
            console.error(`Không tìm thấy phần tử HTML cho stream ${streamId}`);
            return;
        }

        const ctx = canvas.getContext("2d");
        const img = new Image();

        // Xử lý khi ảnh (frame) được tải xong
        img.onload = () => {
            // Đảm bảo canvas có cùng kích thước với ảnh nhận được
            if (canvas.width !== img.width || canvas.height !== img.height) {
                canvas.width = img.width;
                canvas.height = img.height;
            }
            // Vẽ frame lên canvas
            ctx.drawImage(img, 0, 0);
        };
        
        img.onerror = (e) => {
            console.error(`Lỗi tải ảnh Base64 cho stream ${streamId}:`, e);
        };

        // Hàm kết nối WebSocket
        function connectWebSocket() {
            // Lấy giao thức (ws:// hoặc wss://)
            const wsProtocol = window.location.protocol === "https:" ? "wss://" : "ws://";
            const wsUrl = `${wsProtocol}${window.location.host}/ws/${streamId}`;
            console.log(`Đang kết nối tới stream ${streamId} tại: ${wsUrl}`);

            const socket = new WebSocket(wsUrl);

            socket.onopen = () => {
                console.log(`WebSocket đã kết nối cho stream ${streamId}.`);
                dataDisplay.textContent = "Đã kết nối...";
                dataDisplay.style.color = "#4CAF50"; // Màu xanh
            };

            // Hàm chính: Nhận dữ liệu từ server
            socket.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    
                    // Cập nhật ảnh
                    // Thêm tiền tố data URI để trình duyệt hiểu đây là ảnh JPEG base64
                    img.src = "data:image/jpeg;base64," + data.image; 

                    // Cập nhật dữ liệu (ví dụ: bộ đếm)
                    dataDisplay.textContent = `Count: ${data.count}`;

                } catch (e) {
                    console.error(`Lỗi phân tích JSON cho stream ${streamId}:`, e);
                }
            };

            socket.onclose = (event) => {
                console.warn(`WebSocket đã đóng cho stream ${streamId}. Đang thử kết nối lại sau 3 giây...`, event.reason);
                dataDisplay.textContent = "Mất kết nối. Đang thử lại...";
                dataDisplay.style.color = "#FF9800"; // Màu cam
                
                // Tự động kết nối lại
                setTimeout(connectWebSocket, 3000);
            };

            socket.onerror = (error) => {
                console.error(`Lỗi WebSocket cho stream ${streamId}:`, error);
                dataDisplay.textContent = "Lỗi kết nối.";
                dataDisplay.style.color = "#F44336"; // Màu đỏ
                socket.close(); // Đóng kết nối để kích hoạt onclose (và kết nối lại)
            };
        }

        // Bắt đầu kết nối
        connectWebSocket();
    });
});
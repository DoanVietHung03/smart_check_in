document.addEventListener('DOMContentLoaded', function() {
    const videoContainer = document.querySelector('.video-container');
    if (!videoContainer) {
        console.error("Container chính không tìm thấy!");
        return;
    }
    
    // Đọc số lượng stream từ thẻ HTML (được tạo bởi Python)
    const stream_count = parseInt(videoContainer.dataset.streamCount, 10);
    const contexts = new Map(); 

    /**
     * Hàm render danh sách nhận diện vào sidebar
     * Hàm này sẽ xóa sidebar cũ và tạo lại toàn bộ
     */
    const renderIdentitySidebar = (stream_id, identities) => {
        const sidebar = document.getElementById(`identity-sidebar-${stream_id}`);
        if (!sidebar) return;

        // Xóa sạch nội dung sidebar cũ để cập nhật danh sách mới nhất
        sidebar.innerHTML = ''; 

        if (identities.length === 0) {
            sidebar.innerHTML = '<p class="sidebar-empty">Không có ai được nhận diện.</p>';
            return;
        }

        // Tạo thẻ (card) cho mỗi người được nhận diện
        identities.forEach(person => {
            // Tạo div bao ngoài
            const card = document.createElement('div');
            card.className = 'identity-card';

            // Tạo thẻ ảnh thumbnail
            const img = document.createElement('img');
            img.src = "data:image/jpeg;base64," + person.thumb; // Lấy thumbnail base64
            img.className = 'identity-thumb';

            // Tạo thẻ tên
            const nameTag = document.createElement('p');
            nameTag.className = 'identity-name';
            nameTag.innerText = person.name;
            
            // QUAN TRỌNG: Áp dụng màu sắc (đã được server gửi ở định dạng CSS)
            nameTag.style.color = person.color;

            card.appendChild(img);
            card.appendChild(nameTag);
            sidebar.appendChild(card);
        });
    };

    /**
     * Hàm thiết lập kết nối WebSocket cho mỗi stream
     */
    const setupWebSocket = (stream_id) => {
        const canvas = document.getElementById('video-stream-' + stream_id);
        if (!canvas) {
            console.error(`Canvas 'video-stream-${stream_id}' không tìm thấy!`);
            return;
        }
        
        const ctx = canvas.getContext('2d', { alpha: false });
        contexts.set(stream_id, ctx);
        
        const wsProtocol = window.location.protocol === "https:" ? "wss://" : "ws://";
        const wsUrl = `${wsProtocol}${location.host}/ws/${stream_id}`;
        const ws = new WebSocket(wsUrl);

        ws.onmessage = async (ev) => {
            try {
                const data = JSON.parse(ev.data);

                // 1. Vẽ khung hình video chính (như cũ)
                const image = new Image();
                image.onload = () => {
                    if (ctx.canvas.width !== image.width || ctx.canvas.height !== image.height) {
                        ctx.canvas.width = image.width;
                        ctx.canvas.height = image.height;
                    }
                    ctx.drawImage(image, 0, 0);
                };
                image.src = 'data:image/jpeg;base64,' + data.image;

                // 2. LOGIC MỚI: Render sidebar
                // (Server không còn gửi 'count', thay vào đó là 'identities')
                renderIdentitySidebar(data.stream_id, data.identities);

            } catch (e) {
                console.error('Lỗi xử lý tin nhắn:', e);
            }
        };

        ws.onerror = (e) => console.error(`Lỗi WebSocket trên stream ${stream_id}:`, e);
        ws.onclose = () => {
            console.log(`WebSocket cho stream ${stream_id} đã đóng. Đang kết nối lại...`);
            setTimeout(() => setupWebSocket(stream_id), 3000);
        };
    };

    // Tạo kết nối cho mỗi stream
    for (let i = 0; i < stream_count; i++) {
        setupWebSocket(i);
    }
});
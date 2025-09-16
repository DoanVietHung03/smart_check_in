// main.js (ĐÃ SỬA ĐỂ TẢI THUMBNAIL TỪ URL)

document.addEventListener('DOMContentLoaded', function() {
    const videoContainer = document.querySelector('.video-container');
    if (!videoContainer) {
        console.error("Container chính không tìm thấy!");
        return;
    }
    
    const stream_count = parseInt(videoContainer.dataset.streamCount, 10);
    const contexts = new Map(); 

    /**
     * Hàm render danh sách nhận diện vào sidebar
     */
    const renderIdentitySidebar = (stream_id, identities) => {
        const sidebar = document.getElementById(`identity-sidebar-${stream_id}`);
        if (!sidebar) return;

        sidebar.innerHTML = ''; // Xóa sạch nội dung sidebar cũ

        if (identities.length === 0) {
            sidebar.innerHTML = '<p class="sidebar-empty">Không có ai được nhận diện.</p>';
            return;
        }

        identities.forEach(person => {
            const card = document.createElement('div');
            card.className = 'identity-card';

            const img = document.createElement('img');
            
            // --- THAY ĐỔI LỚN DUY NHẤT Ở ĐÂY ---
            // 'person.thumb' giờ là một URL (ví dụ: /gallery_images/John/01.jpg)
            // Nếu nó là null hoặc rỗng, chúng ta không đặt src (nó sẽ hiển thị nền mặc định)
            if (person.thumb) {
                 img.src = person.thumb; 
            }
            // ------------------------------------

            img.className = 'identity-thumb';

            const nameTag = document.createElement('p');
            nameTag.className = 'identity-name';
            nameTag.innerText = person.name;
            nameTag.style.color = person.color;

            card.appendChild(img);
            card.appendChild(nameTag);
            sidebar.appendChild(card);
        });
    };

    /**
     * Hàm thiết lập kết nối WebSocket
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

                // 2. Render sidebar (như cũ)
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
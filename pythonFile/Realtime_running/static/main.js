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

        // 1. Lấy danh sách TÊN hiện đang có trong sidebar
        const existingNames = new Set();
        sidebar.querySelectorAll('.identity-card').forEach(card => {
            existingNames.add(card.dataset.name); // Sử dụng data-name
        });

        // 2. Lấy danh sách TÊN mới từ server
         const newNames = new Set(identities.map(p => p.name));

        // 3. XÓA những card có Tên không còn trong danh sách mới
        sidebar.querySelectorAll('.identity-card').forEach(card => {
            if (!newNames.has(card.dataset.name)) {
                card.remove(); // Xóa người đã rời đi
            }
        });

        // 4. THÊM những người có Tên mới mà chưa có trong sidebar
        identities.forEach(person => {
            // Nếu TÊN này chưa có trong sidebar, chúng ta mới tạo và thêm vào
            if (!existingNames.has(person.name)) {
                const card = document.createElement('div');
                card.className = 'identity-card';
                // Gán tracker_id vào 'data-id' để theo dõi
                card.dataset.name = person.name; // Gán TÊN vào 'data-name'

                const img = document.createElement('img');
                if (person.thumb) {
                    img.src = person.thumb; 
                }
                img.className = 'identity-thumb';

                const nameTag = document.createElement('p');
                nameTag.className = 'identity-name';
                nameTag.innerText = person.name;
                nameTag.style.color = person.color;

                card.appendChild(img);
                card.appendChild(nameTag);
                sidebar.appendChild(card); // Thêm người mới
            }
            // Nếu người đó đã tồn tại (existingIDs.has(personIDStr) == true), chúng ta không làm gì cả
        });
        
        // 5. Xử lý thông báo "rỗng"
        const emptyMsg = sidebar.querySelector('.sidebar-empty');
        if (sidebar.querySelectorAll('.identity-card').length > 0) {
             if (emptyMsg) emptyMsg.remove();
        } else if (!emptyMsg) {
             sidebar.innerHTML = '<p class="sidebar-empty">Không có ai được nhận diện.</p>';
        }
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

                // 1. Vẽ khung hình video chính
                const image = new Image();
                image.onload = () => {
                    if (ctx.canvas.width !== image.width || ctx.canvas.height !== image.height) {
                        ctx.canvas.width = image.width;
                        ctx.canvas.height = image.height;
                    }
                    ctx.drawImage(image, 0, 0);
                };
                image.src = 'data:image/jpeg;base64,' + data.image;

                // 2. Render sidebar (vẫn gọi hàm này, nhưng logic bên trong đã thay đổi)
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
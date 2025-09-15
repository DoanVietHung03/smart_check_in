# db_manager.py (VIẾT LẠI HOÀN TOÀN CHO MYSQL)

import mysql.connector
from mysql.connector.pooling import MySQLConnectionPool
from mysql.connector import errorcode
import time
import logging

# Thiết lập logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('database')

# Lấy cấu hình DB từ biến môi trường (do docker-compose cung cấp)
DB_CONFIG = {
    'host': 'db',
    'port': 3306,
    'user': 'root',
    'password': '12345',
    'database': 'rtsp_db',
}

# Khởi tạo connection pool
POOL_CONFIG = {
    "pool_name": "mypool",
    "pool_size": 5,  # Đủ cho n luồng RTSP
    **DB_CONFIG
}

try:
    db_pool = MySQLConnectionPool(**POOL_CONFIG)
    logger.info("Database connection pool initialized.")
except mysql.connector.Error as err:
    logger.error(f"Error initializing database connection pool: {err}")

def initialize_database(stream_ids: list):
    """
    Hàm này chạy một lần để đảm bảo DB và các bảng tồn tại.
    Tạo một bảng riêng cho mỗi stream_id trong danh sách.
    """
    try:
        # Kết nối không chỉ định database để tạo database
        temp_config = DB_CONFIG.copy()
        temp_config.pop('database', None)
        
        with mysql.connector.connect(**temp_config) as cnx:
            with cnx.cursor() as cur:
                cur.execute(f"CREATE DATABASE IF NOT EXISTS {DB_CONFIG['database']}")
                logger.info(f"Database '{DB_CONFIG['database']}' has been created.")
                
        # Kết nối vào database để tạo bảng cho mỗi stream
        with db_pool.get_connection() as cnx:
            with cnx.cursor() as cur:
                for stream_id in stream_ids:
                    # Tạo tên bảng động, thay thế ký tự không hợp lệ
                    table_name = f"recognition_stream_{stream_id}"
                    cur.execute(f"""
                        CREATE TABLE IF NOT EXISTS {table_name} (
                            id INT AUTO_INCREMENT PRIMARY KEY,
                            timestamp DATETIME,
                            tracker_id INT,
                            name VARCHAR(255),
                            score FLOAT,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """)
                    logger.info(f"Table '{table_name}' ensured to exist.")

                    # Kiểm tra xem chỉ mục đã tồn tại chưa
                    cur.execute(f"""
                        SELECT COUNT(*) 
                        FROM information_schema.statistics 
                        WHERE table_schema = '{DB_CONFIG['database']}'
                        AND table_name = '{table_name}' 
                        AND index_name = 'idx_timestamp'
                    """)
                    if cur.fetchone()[0] == 0:  # Chỉ mục chưa tồn tại
                        cur.execute(f"CREATE INDEX idx_timestamp ON {table_name} (timestamp)")
                        logger.info(f"Index idx_timestamp ensured for table '{table_name}'.")

            cnx.commit()
            logger.info("Database initialized successfully.")
    except mysql.connector.Error as err:
        logger.error(f"Failed to initialize database: {err}")
        raise

def db_insert(recognition_event: dict, stream_id):
    """
    Hàm insert MỘT sự kiện nhận dạng (khi một người được xác định).
    Sử dụng ON DUPLICATE KEY UPDATE để tránh ghi trùng lặp tracker_id.
    """
    if not recognition_event:
        return

    # Tạo tên bảng động
    table_name = f"recognition_stream_{stream_id}"
    
    # Chuyển đổi dữ liệu sang dạng tuple để dùng executemany
    sql = f"""
    INSERT INTO {table_name} (stream_id, tracker_id, name, score, timestamp) 
    VALUES (%s, %s, %s, %s, %s)
    ON DUPLICATE KEY UPDATE
        name = VALUES(name), 
        score = VALUES(score),
        timestamp = VALUES(timestamp)
    """
    data_tuple = (
        recognition_event["stream_id"],
        recognition_event["tracker_id"],
        recognition_event["name"],
        recognition_event["score"],
        recognition_event["timestamp"]
    )
    
    conn = None
    try:
        conn = db_pool.get_connection()
        cur = conn.cursor()
        cur.execute(sql, data_tuple)
        conn.commit()
        
    except mysql.connector.Error as err:
        logger.error(f"Database insert error for table '{table_name}': {err}")
        if cnx:
            cnx.rollback()
    finally:
        if conn and conn.is_connected():
            cur.close()
            conn.close()
# db_manager.py (VIẾT LẠI HOÀN TOÀN CHO MYSQL)

import mysql.connector
from mysql.connector.pooling import MySQLConnectionPool
from mysql.connector import errorcode
import os
import time
import logging

# Lấy cấu hình DB từ biến môi trường (do docker-compose cung cấp)
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'user': os.getenv('DB_USER', 'root'),
    'password': os.getenv('DB_PASS', '12345'),
    'database': os.getenv('DB_NAME', 'rtsp_db'),
}

TABLE_NAME = 'recognition_events'
CREATE_TABLE_QUERY = f"""
CREATE TABLE IF NOT EXISTS `{TABLE_NAME}` (
    `id` INT AUTO_INCREMENT PRIMARY KEY,
    `timestamp` DATETIME DEFAULT CURRENT_TIMESTAMP,
    `stream_id` VARCHAR(50),
    `tracker_id` INT,
    `name` VARCHAR(255),
    `score` FLOAT
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""

# Khởi tạo một Connection Pool toàn cục
# Đây là cách làm đúng để quản lý kết nối trong một ứng dụng đa luồng (như db_worker)
try:
    print(f"Initializing MySQL Connection Pool for host: {DB_CONFIG['host']}...")
    db_pool = MySQLConnectionPool(
        pool_name="fa_pool",
        pool_size=5,  # 5 kết nối sẵn sàng cho db_worker
        **DB_CONFIG
    )
    print("MySQL Connection Pool initialized.")
except mysql.connector.Error as err:
    print(f"Error initializing connection pool: {err}")
    # Nếu DB chưa sẵn sàng, pool sẽ được tạo nhưng không có kết nối. 
    # Hàm initialize_database() sẽ xử lý việc kết nối lại.
    db_pool = None 

def initialize_database():
    """
    Hàm này được gọi bởi server:app khi khởi động.
    Nó sẽ kết nối, tạo bảng nếu chưa có.
    Nó sẽ thử lại nhiều lần vì app có thể khởi động nhanh hơn DB.
    """
    global db_pool
    print("Attempting to initialize database schema...")
    conn = None
    retries = 10
    wait_time = 5 # Chờ 5 giây giữa các lần thử

    for i in range(retries):
        try:
            # Nếu pool bị lỗi khi khởi tạo, thử tạo lại
            if db_pool is None:
                 db_pool = MySQLConnectionPool(
                    pool_name="fa_pool",
                    pool_size=5,
                    **DB_CONFIG
                )
                 print("Connection Pool re-initialized during startup.")

            # Lấy 1 kết nối từ pool để kiểm tra và tạo bảng
            conn = db_pool.get_connection()
            print(f"Attempt {i+1}/{retries}: MySQL DB connected successfully.")
            cursor = conn.cursor()
            cursor.execute(CREATE_TABLE_QUERY)
            conn.commit()
            print(f"Table '{TABLE_NAME}' ensured to exist.")
            return True # Thành công

        except mysql.connector.Error as err:
            if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
                print("Lỗi xác thực (sai user/pass). Sẽ không thử lại.")
                break
            elif err.errno == errorcode.ER_BAD_DB_ERROR:
                print(f"Database '{DB_CONFIG['database']}' không tồn tại (chưa được tạo bởi service 'db'?). Sẽ không thử lại.")
                break
            else:
                print(f"Attempt {i+1}/{retries}: DB connection failed (service có thể đang khởi động...): {err}")
                time.sleep(wait_time)
        finally:
            if conn and conn.is_connected():
                cursor.close()
                conn.close() # Trả kết nối về pool

    print("FATAL: Could not initialize database after all retries. Exiting.")
    return False
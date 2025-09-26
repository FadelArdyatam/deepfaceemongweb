#!/usr/bin/env python3
"""
Script untuk membuat tabel emotion_aggregations
"""

import pymysql
from config import Config
from datetime import datetime

def create_aggregation_table():
    """Buat tabel emotion_aggregations"""
    try:
        # Koneksi langsung ke database
        connection = pymysql.connect(
            host=Config.DB_HOST,
            user=Config.DB_USERNAME,
            password=Config.DB_PASSWORD,
            database=Config.DB_NAME,
            charset='utf8mb4'
        )
        
        with connection.cursor() as cursor:
            # Cek apakah tabel sudah ada
            cursor.execute("SHOW TABLES LIKE 'emotion_aggregations'")
            if cursor.fetchone():
                print("✅ Tabel emotion_aggregations sudah ada")
                return True
            
            # Buat tabel
            create_table_sql = """
            CREATE TABLE emotion_aggregations (
                id INT AUTO_INCREMENT PRIMARY KEY,
                teacher_id INT NOT NULL,
                date DATE NOT NULL,
                emotion VARCHAR(20) NOT NULL,
                count INT DEFAULT 0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                UNIQUE KEY unique_teacher_date_emotion (teacher_id, date, emotion),
                FOREIGN KEY (teacher_id) REFERENCES users(id) ON DELETE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci
            """
            
            cursor.execute(create_table_sql)
            connection.commit()
            print("✅ Tabel emotion_aggregations berhasil dibuat")
            
            return True
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    finally:
        if 'connection' in locals():
            connection.close()

if __name__ == "__main__":
    success = create_aggregation_table()
    if success:
        print("🎉 Tabel agregasi berhasil dibuat!")
    else:
        print("💥 Gagal membuat tabel agregasi!")
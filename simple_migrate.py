#!/usr/bin/env python3
"""
Script sederhana untuk memastikan student_id di emotion_sessions nullable
"""

import pymysql
import os
from config import Config

def migrate_student_id_nullable():
    """Pastikan student_id di emotion_sessions bisa NULL"""
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
            # Cek struktur tabel saat ini
            cursor.execute("SHOW CREATE TABLE emotion_sessions")
            result = cursor.fetchone()
            create_table_sql = result[1]
            print("Struktur tabel emotion_sessions saat ini:")
            print(create_table_sql)
            
            # Cek apakah student_id sudah nullable
            if "`student_id` int(11) DEFAULT NULL" in create_table_sql or "`student_id` int DEFAULT NULL" in create_table_sql:
                print("✅ student_id sudah nullable")
                return True
            
            # Jika belum nullable, ubah menjadi nullable
            print("🔄 Mengubah student_id menjadi nullable...")
            cursor.execute("ALTER TABLE emotion_sessions MODIFY COLUMN student_id INT NULL")
            connection.commit()
            print("✅ student_id berhasil diubah menjadi nullable")
            
            # Verifikasi perubahan
            cursor.execute("SHOW CREATE TABLE emotion_sessions")
            result = cursor.fetchone()
            create_table_sql = result[1]
            print("Struktur tabel emotion_sessions setelah perubahan:")
            print(create_table_sql)
            
            return True
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    finally:
        if 'connection' in locals():
            connection.close()

if __name__ == "__main__":
    success = migrate_student_id_nullable()
    if success:
        print("🎉 Migrasi berhasil!")
    else:
        print("💥 Migrasi gagal!")
#!/usr/bin/env python3
"""
Script untuk memastikan student_id di emotion_sessions nullable
"""

from app import app, db
from sqlalchemy import text

def migrate_student_id_nullable():
    """Pastikan student_id di emotion_sessions bisa NULL"""
    with app.app_context():
        try:
            # Cek struktur tabel saat ini
            result = db.engine.execute(text("SHOW CREATE TABLE emotion_sessions"))
            create_table_sql = result.fetchone()[1]
            print("Struktur tabel emotion_sessions saat ini:")
            print(create_table_sql)
            
            # Cek apakah student_id sudah nullable
            if "`student_id` int(11) DEFAULT NULL" in create_table_sql or "`student_id` int DEFAULT NULL" in create_table_sql:
                print("✅ student_id sudah nullable")
                return True
            
            # Jika belum nullable, ubah menjadi nullable
            print("🔄 Mengubah student_id menjadi nullable...")
            db.engine.execute(text("ALTER TABLE emotion_sessions MODIFY COLUMN student_id INT NULL"))
            db.session.commit()
            print("✅ student_id berhasil diubah menjadi nullable")
            
            # Verifikasi perubahan
            result = db.engine.execute(text("SHOW CREATE TABLE emotion_sessions"))
            create_table_sql = result.fetchone()[1]
            print("Struktur tabel emotion_sessions setelah perubahan:")
            print(create_table_sql)
            
            return True
            
        except Exception as e:
            print(f"❌ Error: {e}")
            db.session.rollback()
            return False

if __name__ == "__main__":
    success = migrate_student_id_nullable()
    if success:
        print("🎉 Migrasi berhasil!")
    else:
        print("💥 Migrasi gagal!")
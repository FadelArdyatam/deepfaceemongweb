#!/usr/bin/env python3
"""
Migration script untuk menambahkan kolom photo_path dan notes ke tabel students
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app, db
from sqlalchemy import text

def add_student_fields():
    """Tambahkan kolom photo_path dan notes ke tabel students"""
    with app.app_context():
        try:
            # Cek apakah kolom sudah ada
            with db.engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT COLUMN_NAME 
                    FROM INFORMATION_SCHEMA.COLUMNS 
                    WHERE TABLE_NAME = 'students' 
                    AND COLUMN_NAME IN ('photo_path', 'notes')
                """))
                
                existing_columns = [row[0] for row in result]
                
                # Tambahkan kolom photo_path jika belum ada
                if 'photo_path' not in existing_columns:
                    print("Menambahkan kolom photo_path...")
                    conn.execute(text("""
                        ALTER TABLE students 
                        ADD COLUMN photo_path VARCHAR(255) NULL
                    """))
                    conn.commit()
                    print("✓ Kolom photo_path berhasil ditambahkan")
                else:
                    print("✓ Kolom photo_path sudah ada")
                
                # Tambahkan kolom notes jika belum ada
                if 'notes' not in existing_columns:
                    print("Menambahkan kolom notes...")
                    conn.execute(text("""
                        ALTER TABLE students 
                        ADD COLUMN notes TEXT NULL
                    """))
                    conn.commit()
                    print("✓ Kolom notes berhasil ditambahkan")
                else:
                    print("✓ Kolom notes sudah ada")
            
            print("Migration berhasil!")
            
        except Exception as e:
            print(f"Error migration: {e}")
            return False
    
    return True

if __name__ == "__main__":
    add_student_fields()
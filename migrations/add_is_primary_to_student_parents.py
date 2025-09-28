"""
Migration untuk menambahkan kolom is_primary ke tabel student_parents
"""
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text

def upgrade(db):
    """Menambahkan kolom is_primary ke tabel student_parents"""
    try:
        # Tambahkan kolom is_primary dengan default value False
        db.session.execute(text("""
            ALTER TABLE student_parents 
            ADD COLUMN is_primary BOOLEAN DEFAULT FALSE
        """))
        db.session.commit()
        print("✅ Kolom is_primary berhasil ditambahkan ke tabel student_parents")
    except Exception as e:
        db.session.rollback()
        print(f"❌ Error menambahkan kolom is_primary: {e}")
        raise e

def downgrade(db):
    """Menghapus kolom is_primary dari tabel student_parents"""
    try:
        db.session.execute(text("""
            ALTER TABLE student_parents 
            DROP COLUMN is_primary
        """))
        db.session.commit()
        print("✅ Kolom is_primary berhasil dihapus dari tabel student_parents")
    except Exception as e:
        db.session.rollback()
        print(f"❌ Error menghapus kolom is_primary: {e}")
        raise e

if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from app import app, db
    with app.app_context():
        upgrade(db)
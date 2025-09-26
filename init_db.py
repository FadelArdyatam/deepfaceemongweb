#!/usr/bin/env python3
"""
Script untuk inisialisasi database dan data sample
Jalankan dengan: python init_db.py
"""

from app import app, db
from models import User, Student, StudentTeacher, StudentParent
from werkzeug.security import generate_password_hash
from datetime import date

def init_database():
    """Inisialisasi database dan buat tabel"""
    with app.app_context():
        print("🗄️ Membuat tabel database...")
        db.create_all()
        print("✅ Tabel database berhasil dibuat!")

def create_sample_data():
    """Buat data sample untuk testing"""
    with app.app_context():
        print("👥 Membuat data sample...")
        
        # Cek apakah sudah ada data
        if User.query.first():
            print("⚠️ Data sudah ada, skip pembuatan data sample")
            return
        
        # Buat admin
        admin = User(
            username='admin',
            email='admin@emotiondetection.com',
            role='admin',
            full_name='Administrator',
            phone='081234567890'
        )
        admin.set_password('admin123')
        db.session.add(admin)
        
        # Buat guru
        guru1 = User(
            username='guru1',
            email='guru1@school.com',
            role='guru',
            full_name='Budi Santoso',
            phone='081234567891'
        )
        guru1.set_password('guru123')
        db.session.add(guru1)
        
        guru2 = User(
            username='guru2',
            email='guru2@school.com',
            role='guru',
            full_name='Siti Rahayu',
            phone='081234567892'
        )
        guru2.set_password('guru123')
        db.session.add(guru2)
        
        # Buat orang tua
        parent1 = User(
            username='parent1',
            email='parent1@email.com',
            role='orang_tua',
            full_name='Ahmad Wijaya',
            phone='081234567893'
        )
        parent1.set_password('parent123')
        db.session.add(parent1)
        
        parent2 = User(
            username='parent2',
            email='parent2@email.com',
            role='orang_tua',
            full_name='Sari Dewi',
            phone='081234567894'
        )
        parent2.set_password('parent123')
        db.session.add(parent2)
        
        # Buat siswa
        student1 = Student(
            student_code='S001',
            full_name='Andi Pratama',
            class_name='7A',
            birth_date=date(2010, 5, 15),
            photo_path='students/andi.jpg'
        )
        db.session.add(student1)
        
        student2 = Student(
            student_code='S002',
            full_name='Bella Putri',
            class_name='7A',
            birth_date=date(2010, 8, 22),
            photo_path='students/bella.jpg'
        )
        db.session.add(student2)
        
        student3 = Student(
            student_code='S003',
            full_name='Candra Kurniawan',
            class_name='7B',
            birth_date=date(2010, 3, 10),
            photo_path='students/candra.jpg'
        )
        db.session.add(student3)
        
        student4 = Student(
            student_code='S004',
            full_name='Dina Sari',
            class_name='7B',
            birth_date=date(2010, 12, 5),
            photo_path='students/dina.jpg'
        )
        db.session.add(student4)
        
        # Commit untuk mendapatkan ID
        db.session.commit()
        
        # Buat relasi guru-siswa
        # Guru1 mengajar kelas 7A
        st1 = StudentTeacher(
            student_id=student1.id,
            teacher_id=guru1.id,
            subject='Matematika'
        )
        db.session.add(st1)
        
        st2 = StudentTeacher(
            student_id=student2.id,
            teacher_id=guru1.id,
            subject='Matematika'
        )
        db.session.add(st2)
        
        # Guru2 mengajar kelas 7B
        st3 = StudentTeacher(
            student_id=student3.id,
            teacher_id=guru2.id,
            subject='Bahasa Indonesia'
        )
        db.session.add(st3)
        
        st4 = StudentTeacher(
            student_id=student4.id,
            teacher_id=guru2.id,
            subject='Bahasa Indonesia'
        )
        db.session.add(st4)
        
        # Buat relasi orang tua-siswa
        # Parent1 adalah ayah dari Andi
        sp1 = StudentParent(
            student_id=student1.id,
            parent_id=parent1.id,
            relationship='ayah',
            is_primary=True
        )
        db.session.add(sp1)
        
        # Parent2 adalah ibu dari Bella
        sp2 = StudentParent(
            student_id=student2.id,
            parent_id=parent2.id,
            relationship='ibu',
            is_primary=True
        )
        db.session.add(sp2)
        
        # Parent1 juga wali dari Candra
        sp3 = StudentParent(
            student_id=student3.id,
            parent_id=parent1.id,
            relationship='wali',
            is_primary=False
        )
        db.session.add(sp3)
        
        db.session.commit()
        
        print("✅ Data sample berhasil dibuat!")
        print("\n📋 Akun untuk testing:")
        print("Admin: admin / admin123")
        print("Guru 1: guru1 / guru123")
        print("Guru 2: guru2 / guru123")
        print("Orang Tua 1: parent1 / parent123")
        print("Orang Tua 2: parent2 / parent123")

def main():
    """Main function"""
    print("🚀 Memulai inisialisasi database...")
    
    try:
        init_database()
        create_sample_data()
        print("\n🎉 Inisialisasi database selesai!")
        print("💡 Jalankan aplikasi dengan: python app.py")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())
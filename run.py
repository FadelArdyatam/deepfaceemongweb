#!/usr/bin/env python3
"""
Script untuk menjalankan aplikasi Emotion Detection System
Jalankan dengan: python run.py
"""

import os
import sys
from app import app, db

def check_environment():
    """Cek environment variables yang diperlukan"""
    required_vars = [
        'DB_HOST', 'DB_USERNAME', 'DB_PASSWORD', 'DB_NAME',
        'JWT_SECRET_KEY', 'SECRET_KEY'
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print("❌ Environment variables yang diperlukan tidak ditemukan:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\n💡 Buat file .env dengan konfigurasi yang diperlukan.")
        print("   Lihat README_SETUP.md untuk panduan lengkap.")
        return False
    
    return True

def main():
    """Main function"""
    print("🚀 Memulai Emotion Detection System...")
    
    # Cek environment
    if not check_environment():
        sys.exit(1)
    
    try:
        # Inisialisasi database
        with app.app_context():
            print("🗄️ Menginisialisasi database...")
            db.create_all()
            print("✅ Database siap!")
        
        # Jalankan aplikasi
        print("🌐 Menjalankan server...")
        print("📱 Akses aplikasi di: http://localhost:5000")
        print("🔑 Login dengan akun sample:")
        print("   Admin: admin / admin123")
        print("   Guru: guru1 / guru123")
        print("   Orang Tua: parent1 / parent123")
        print("\n⏹️ Tekan Ctrl+C untuk menghentikan server")
        
        port = int(os.environ.get('PORT', 5000))
        app.run(host='0.0.0.0', port=port, debug=True)
        
    except KeyboardInterrupt:
        print("\n👋 Server dihentikan oleh user")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
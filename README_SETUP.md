# Setup Emotion Detection System dengan Autentikasi

## 🚀 Instruksi Setup

### 1. Install Dependencies
```bash
pip install -r requirement.txt
```

### 2. Setup Database MySQL
1. Buat database MySQL:
```sql
CREATE DATABASE emotion_detection_db;
```

2. Buat file `.env` di root project dengan konfigurasi:
```env
# Database Configuration
DB_HOST=localhost
DB_PORT=3306
DB_USERNAME=root
DB_PASSWORD=your_mysql_password
DB_NAME=emotion_detection_db

# JWT Configuration
JWT_SECRET_KEY=your-super-secret-jwt-key-change-this-in-production-12345
JWT_ACCESS_TOKEN_EXPIRES=86400

# Flask Configuration
FLASK_ENV=development
FLASK_DEBUG=True
SECRET_KEY=your-super-secret-flask-key-change-this-in-production-67890

# API Configuration
API_BASE_URL=http://localhost:5000
```

### 3. Inisialisasi Database
```bash
python init_db.py
```

### 4. Jalankan Aplikasi
```bash
python app.py
```

### 5. Akses Aplikasi
- Buka browser ke: http://localhost:5000
- Login dengan akun sample:
  - **Admin**: admin / admin123
  - **Guru 1**: guru1 / guru123
  - **Guru 2**: guru2 / guru123
  - **Orang Tua 1**: parent1 / parent123
  - **Orang Tua 2**: parent2 / parent123

## 📋 Fitur yang Sudah Diimplementasi

### ✅ Autentikasi & Authorization
- Login/Register dengan validasi lengkap
- JWT token authentication dengan expiration
- Role-based access control (Guru, Orang Tua, Admin)
- Password hashing dengan bcrypt
- Middleware untuk proteksi endpoint

### ✅ Database Schema (MySQL)
- Tabel Users (guru, orang tua, admin)
- Tabel Students (data siswa)
- Tabel StudentTeachers (relasi guru-siswa)
- Tabel StudentParents (relasi orang tua-siswa)
- Tabel EmotionSessions (sesi deteksi emosi)
- Tabel EmotionLogs (log deteksi emosi)
- Tabel Reports (laporan)
- Indexing untuk performa optimal

### ✅ Dashboard Guru
- Real-time emotion detection (existing feature)
- Manajemen sesi deteksi emosi
- Daftar siswa yang diajar
- Statistik dashboard dengan grafik
- Live monitoring emotion siswa
- Session controls (start/stop/pause)

### ✅ Dashboard Orang Tua
- Lihat data anak-anak
- Laporan emosi dengan timeline
- Grafik perkembangan emosi
- Insight dan tips parenting
- Download laporan (coming soon)

### ✅ Dashboard Admin
- Overview sistem lengkap
- Manajemen user (CRUD)
- Manajemen siswa (CRUD)
- Monitoring semua sesi
- Analytics sistem
- System monitoring

### ✅ API Endpoints Lengkap
**Authentication:**
- `POST /api/auth/register` - Register user
- `POST /api/auth/login` - Login
- `GET /api/auth/profile` - Profil user
- `PUT /api/auth/profile` - Update profil
- `POST /api/auth/change-password` - Ganti password

**Dashboard:**
- `GET /api/dashboard/guru/stats` - Statistik guru
- `GET /api/dashboard/parent/stats` - Statistik orang tua
- `GET /api/dashboard/admin/stats` - Statistik admin

**Student Management:**
- `GET /api/students` - Daftar siswa (guru/admin)
- `GET /api/parent/children` - Daftar anak (orang tua)

**Session Management:**
- `POST /api/sessions` - Buat sesi baru
- `POST /api/sessions/{id}/stop` - Hentikan sesi

**Reports:**
- `GET /api/parent/reports/{child_id}` - Laporan emosi anak

**Admin:**
- `GET /api/admin/users` - Semua users
- `GET /api/admin/students` - Semua siswa
- `GET /api/admin/sessions` - Semua sesi

## 🔄 Fitur yang Masih Dalam Pengembangan

### ⏳ Integrasi Deteksi Emosi
- Integrasi real-time detection dengan session management
- Log emosi otomatis ke database saat sesi aktif
- Real-time analytics dan notifikasi

### ⏳ Laporan & Analytics Advanced
- Generate laporan PDF/Excel otomatis
- Advanced analytics dengan machine learning
- Predictive analysis untuk emosi
- Export data dalam berbagai format

### ⏳ Fitur Tambahan
- Chat/messaging antara guru dan orang tua
- Notifikasi real-time (email/push)
- Mobile app (React Native/Flutter)
- Advanced user management (bulk operations)
- Backup/restore database
- System monitoring dan alerting

## 🛠️ Struktur Project

```
RealtimeEmotionDetection/
├── app.py                 # Main Flask application
├── models.py              # Database models
├── auth.py                # Authentication system
├── config.py              # Configuration
├── init_db.py             # Database initialization
├── requirement.txt        # Dependencies
├── templates/
│   ├── login.html         # Login page
│   ├── dashboard_guru.html # Dashboard guru
│   ├── dashboard_parent.html # Dashboard orang tua (coming soon)
│   └── dashboard_admin.html # Dashboard admin (coming soon)
├── static/
│   └── style.css          # CSS styles
└── uploads/               # Upload directory
```

## 🔧 Troubleshooting

### Error Database Connection
- Pastikan MySQL server running
- Cek konfigurasi di file `.env`
- Pastikan database `emotion_detection_db` sudah dibuat

### Error Import Module
- Pastikan semua dependencies terinstall: `pip install -r requirement.txt`
- Pastikan Python path sudah benar

### Error JWT Token
- Cek konfigurasi `JWT_SECRET_KEY` di file `.env`
- Pastikan token tidak expired

## 📞 Support

Jika ada masalah atau pertanyaan, silakan buat issue di repository atau hubungi developer.
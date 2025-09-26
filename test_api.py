#!/usr/bin/env python3
"""
Script untuk testing API endpoints
Jalankan dengan: BASE_URL=http://localhost:5000 python test_api.py
"""

import requests
import json
import time
import os
from urllib.parse import urljoin
import base64

try:
    import numpy as np
    import cv2
except Exception:
    np = None
    cv2 = None

def wait_for_server(base_url: str, timeout_seconds: int = 25, interval_seconds: float = 0.5) -> bool:
    """Tunggu hingga server merespons salah satu endpoint sederhana."""
    start = time.time()
    probe_paths = ['/config', '/api/auth/profile', '/']
    while time.time() - start < timeout_seconds:
        for path in probe_paths:
            try:
                res = requests.get(urljoin(base_url, path), timeout=2)
                if res.status_code < 500:
                    return True
            except Exception:
                pass
        time.sleep(interval_seconds)
    return False

# Base URL dari ENV (default localhost:5000)
BASE_URL = os.getenv("BASE_URL", "http://localhost:5000").rstrip('/')


def make_dummy_image_base64(w=320, h=240, color=(0, 0, 0)):
    if np is None or cv2 is None:
        # fallback: 1x1 blank jpeg
        return base64.b64encode(b"\xff\xd8\xff\xd9").decode('utf-8')
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :] = color
    ok, buf = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
    if not ok:
        return base64.b64encode(b"\xff\xd8\xff\xd9").decode('utf-8')
    return base64.b64encode(buf.tobytes()).decode('utf-8')


def test_api():
    """Test semua API endpoints"""
    print("🧪 Memulai testing API...")

    # Tunggu server siap
    print(f"⏳ Menunggu server siap di {BASE_URL}...")
    if not wait_for_server(BASE_URL):
        print("❌ Server tidak bisa dihubungi. Pastikan app berjalan di port yang benar atau set ENV BASE_URL.")
        return

    # Test data
    test_user = {
        "username": "testuser",
        "email": "test@example.com",
        "password": "Testpass123",
        "role": "guru",
        "full_name": "Test User",
        "phone": "081234567890"
    }

    # 1. Test Register
    print("\n1. Testing Register...")
    token = None
    try:
        response = requests.post(f"{BASE_URL}/api/auth/register", json=test_user, timeout=10)
        if response.status_code == 201:
            print("✅ Register berhasil")
        elif response.status_code == 409:
            print("ℹ️  User sudah ada (409), lanjutkan ke login")
        else:
            try:
                print(f"❌ Register gagal: {response.status_code} {response.json()}")
            except Exception:
                print(f"❌ Register gagal: {response.status_code} {response.text}")
    except Exception as e:
        print(f"❌ Error register: {e}")

    # 2. Test Login (gunakan akun testuser yang barusan diregistrasi)
    print("\n2. Testing Login...")
    try:
        response = requests.post(f"{BASE_URL}/api/auth/login",json={"username": test_user["username"], "password": test_user["password"]}, timeout=10)
        if response.status_code == 200:
            data = response.json()
            token = data['access_token']
            user = data['user']
            print("✅ Login berhasil")
            print(f"Token: {token[:50]}...")
        else:
            try:
                print(f"❌ Login gagal: {response.status_code} {response.json()}")
            except Exception:
                print(f"❌ Login gagal: {response.status_code} {response.text}")
            return
    except Exception as e:
        print(f"❌ Error login: {e}")
        return

    # Headers dengan token
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    # 3. Config & Camera source (set webcam)
    print("\n3. Testing Camera Source (webcam)...")
    try:
        r = requests.post(f"{BASE_URL}/camera/source", json={"source":"webcam"}, timeout=10)
        print("✅ Set camera source (webcam)" if r.ok else f"❌ Set camera source gagal: {r.status_code} {r.text}")
    except Exception as e:
        print(f"❌ Error camera/source: {e}")

    # 4. Dashboard Stats (Guru)
    print("\n4. Testing Dashboard Stats...")
    try:
        response = requests.get(f"{BASE_URL}/api/dashboard/guru/stats", headers=headers, timeout=10)
        if response.status_code == 200:
            print("✅ Dashboard stats berhasil")
        else:
            try:
                print(f"❌ Dashboard stats gagal: {response.status_code} {response.json()}")
            except Exception:
                print(f"❌ Dashboard stats gagal: {response.status_code} {response.text}")
    except Exception as e:
        print(f"❌ Error dashboard stats: {e}")

    # 5. Students: create + upload face (opsional jika file tersedia)
    print("\n5. Testing Students API (create)...")
    try:
        sdata = {
            "student_code": "SIS001",
            "full_name": "Andi Saputra",
            "class_name": "7A",
            "birth_date": "2012-03-05",
            "subject": "Umum"
        }
        response = requests.post(f"{BASE_URL}/api/students", headers=headers, data=None, json=sdata, timeout=15)
        if response.status_code in (201, 409):
            print("✅ Create student OK (201/409)")
        else:
            print(f"❌ Create student gagal: {response.status_code} {response.text}")
    except Exception as e:
        print(f"❌ Error create student: {e}")

    # 6. Create Session (class mode) + list active
    print("\n6. Testing Create Session & Active Sessions...")
    session_id = None
    try:
        session_data = {"session_name": "Test Session", "student_id": 0, "notes": "Testing"}
        response = requests.post(f"{BASE_URL}/api/sessions", json=session_data, headers=headers, timeout=10)
        if response.status_code == 201:
            print("✅ Create session berhasil")
            session = response.json(); session_id = session.get('id')
        else:
            print(f"❌ Create session gagal: {response.status_code} {response.text}")
        ra = requests.get(f"{BASE_URL}/api/sessions/active", headers=headers, timeout=10)
        print("✅ List active sessions" if ra.ok else f"❌ Active sessions gagal: {ra.status_code} {ra.text}")
    except Exception as e:
        print(f"❌ Error create/list session: {e}")

    # 7. Analyze Emotion (dummy image) - with teacher_id/session_id
    print("\n7. Testing analyze_emotion (dummy)...")
    try:
        img64 = make_dummy_image_base64()
        payload = {"image": img64}
        if session_id:
            payload["session_id"] = session_id
        else:
            payload["teacher_id"] = user.get('id')
        response = requests.post(f"{BASE_URL}/analyze_emotion", json=payload, timeout=15)
        if response.ok:
            print("✅ analyze_emotion OK")
        else:
            print(f"❌ analyze_emotion gagal: {response.status_code} {response.text}")
    except Exception as e:
        print(f"❌ Error analyze_emotion: {e}")

    # 8. Daily Summary & Stats
    print("\n8. Testing Daily Summary & Dashboard Stats...")
    try:
        rs = requests.get(f"{BASE_URL}/api/dashboard/guru/stats", headers=headers, timeout=10)
        rds = requests.get(f"{BASE_URL}/api/dashboard/guru/daily-summary", headers=headers, timeout=10)
        print("✅ Stats OK" if rs.ok else f"❌ Stats gagal: {rs.status_code} {rs.text}")
        print("✅ Daily Summary OK" if rds.ok else f"❌ Daily Summary gagal: {rds.status_code} {rds.text}")
    except Exception as e:
        print(f"❌ Error summary/stats: {e}")

    # 9. Stop session (single & bulk)
    print("\n9. Testing Stop Session...")
    try:
        if session_id:
            rstop = requests.post(f"{BASE_URL}/api/sessions/{session_id}/stop", headers=headers, timeout=10)
            print("✅ Stop session OK" if rstop.ok else f"❌ Stop session gagal: {rstop.status_code} {rstop.text}")
        rbulk = requests.post(f"{BASE_URL}/api/sessions/0/stop?all=true", headers=headers, timeout=10)
        print("✅ Stop all OK" if rbulk.ok else f"❌ Stop all gagal: {rbulk.status_code} {rbulk.text}")
    except Exception as e:
        print(f"❌ Error stop session: {e}")

    print("\n🎉 Testing selesai!")

if __name__ == "__main__":
    test_api()
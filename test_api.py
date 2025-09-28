import requests
import json
import time
import sys
from typing import Optional, Dict, Any

class APITester:
    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url
        self.access_token: Optional[str] = None
        self.session = requests.Session()
        
    def login(self, username: str, password: str) -> bool:
        """Login dan dapatkan JWT token"""
        try:
            login_data = {
                "username": username,
                "password": password
            }
            
            response = self.session.post(
                f"{self.base_url}/api/auth/login",
                json=login_data,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                self.access_token = data.get('access_token')
                print(f"✅ Login berhasil sebagai: {data.get('user', {}).get('username', 'Unknown')}")
                return True
            else:
                error_data = response.json() if response.headers.get('content-type', '').startswith('application/json') else {}
                print(f"❌ Login gagal: {error_data.get('error', 'Unknown error')}")
                return False
                
        except requests.exceptions.ConnectionError:
            print("❌ Tidak bisa terhubung ke server. Pastikan aplikasi sudah berjalan di port 5000")
            return False
        except requests.exceptions.Timeout:
            print("❌ Timeout saat login. Server tidak merespons")
            return False
        except Exception as e:
            print(f"❌ Error saat login: {e}")
            return False
    
    def get_headers(self) -> Dict[str, str]:
        """Dapatkan headers dengan JWT token"""
        headers = {"Content-Type": "application/json"}
        if self.access_token:
            headers["Authorization"] = f"Bearer {self.access_token}"
        return headers
    
    def test_link_parent_student(self, parent_id: int, student_code: str, relationship: str = "ayah", is_primary: bool = True) -> bool:
        """Test API endpoint untuk link parent-student"""
        try:
            data = {
                "student_code": student_code,
                "relationship": relationship,
                "is_primary": is_primary
            }
            
            response = self.session.post(
                f"{self.base_url}/api/parents/{parent_id}/link-student",
                json=data,
                headers=self.get_headers(),
                timeout=10
            )
            
            print(f"Status Code: {response.status_code}")
            
            # Parse response
            try:
                response_data = response.json()
                print(f"Response: {json.dumps(response_data, indent=2, ensure_ascii=False)}")
            except:
                print(f"Response (raw): {response.text}")
            
            # Handle different status codes
            if response.status_code == 201:
                print("✅ Relasi parent-student berhasil dibuat!")
                return True
            elif response.status_code == 400:
                print("❌ Bad Request - Data yang dikirim tidak valid")
                return False
            elif response.status_code == 401:
                print("❌ Unauthorized - Token invalid atau expired")
                return False
            elif response.status_code == 403:
                print("❌ Forbidden - Access denied atau role tidak sesuai")
                return False
            elif response.status_code == 404:
                print("❌ Not Found - Parent atau student tidak ditemukan")
                return False
            elif response.status_code == 500:
                print("❌ Internal Server Error - Terjadi kesalahan di server")
                return False
            else:
                print(f"❌ Error dengan status code: {response.status_code}")
                return False
                
        except requests.exceptions.ConnectionError:
            print("❌ Tidak bisa terhubung ke server")
            return False
        except requests.exceptions.Timeout:
            print("❌ Timeout - Server tidak merespons")
            return False
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    def test_without_auth(self, parent_id: int, student_code: str) -> bool:
        """Test API endpoint tanpa authentication untuk melihat error handling"""
        try:
            data = {
                "student_code": student_code,
                "relationship": "ayah",
                "is_primary": True
            }
            
            response = self.session.post(
                f"{self.base_url}/api/parents/{parent_id}/link-student",
                json=data,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            print(f"Status Code (tanpa auth): {response.status_code}")
            
            try:
                response_data = response.json()
                print(f"Response (tanpa auth): {json.dumps(response_data, indent=2, ensure_ascii=False)}")
            except:
                print(f"Response (raw, tanpa auth): {response.text}")
            
            if response.status_code == 401:
                print("✅ Error handling JWT bekerja dengan benar - 401 Unauthorized")
                return True
            else:
                print(f"❌ Expected 401, got {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Error testing without auth: {e}")
            return False

def main():
    print("🚀 Memulai API Testing dengan Error Handling yang Diperbaiki")
    print("=" * 60)
    
    # Tunggu aplikasi startup
    print("⏳ Menunggu aplikasi startup...")
    time.sleep(3)
    
    # Inisialisasi tester
    tester = APITester()
    
    # Test 1: Login
    print("\n📝 Test 1: Login")
    print("-" * 30)
    if not tester.login("admin", "admin123"):  # Ganti dengan credentials yang benar
        print("❌ Tidak bisa melanjutkan tanpa token")
        sys.exit(1)
    
    # Test 2: Test tanpa authentication
    print("\n📝 Test 2: Test tanpa authentication")
    print("-" * 40)
    tester.test_without_auth(4, "Fadel")
    
    # Test 3: Test dengan authentication
    print("\n📝 Test 3: Test dengan authentication")
    print("-" * 40)
    success = tester.test_link_parent_student(4, "Fadel", "ayah", True)
    
    # Test 4: Test dengan data invalid
    print("\n📝 Test 4: Test dengan data invalid")
    print("-" * 40)
    tester.test_link_parent_student(4, "", "ayah", True)  # student_code kosong
    
    # Test 5: Test dengan parent_id yang tidak ada
    print("\n📝 Test 5: Test dengan parent_id yang tidak ada")
    print("-" * 40)
    tester.test_link_parent_student(99999, "Fadel", "ayah", True)
    
    print("\n" + "=" * 60)
    print("🏁 Testing selesai!")

if __name__ == "__main__":
    main()
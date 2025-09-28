#!/usr/bin/env python3
import requests
import json

def test_child_details():
    try:
        print('=== TESTING CHILD DETAILS & EMOTION CHART ===')
        
        # Login
        login_data = {'username': 'parent1', 'password': 'parent123'}
        response = requests.post('http://localhost:5000/api/auth/login', json=login_data)
        
        if response.status_code == 200:
            data = response.json()
            token = data['access_token']
            print('✅ Login berhasil')
            
            headers = {'Authorization': f'Bearer {token}'}
            
            # Test detail anak (ID: 1 - Andi Pratama)
            print('\n--- Testing Child Details API ---')
            child_response = requests.get('http://localhost:5000/api/parent/children/1', headers=headers)
            print(f'Child Details Status: {child_response.status_code}')
            
            if child_response.status_code == 200:
                child_data = child_response.json()
                print('✅ Child Details berhasil')
                print(f'  - Nama: {child_data.get("full_name", "N/A")}')
                print(f'  - Photo Path: {child_data.get("photo_path", "N/A")}')
                print(f'  - Teacher: {child_data.get("teacher_name", "N/A")}')
                print(f'  - Total Detections: {child_data.get("total_detections", 0)}')
                print(f'  - Last Emotion: {child_data.get("last_emotion", "N/A")}')
            else:
                print(f'❌ Child Details gagal: {child_response.text[:200]}')
            
            # Test grafik emosi anak
            print('\n--- Testing Child Emotions API ---')
            emotions_response = requests.get('http://localhost:5000/api/parent/children/1/emotions', headers=headers)
            print(f'Child Emotions Status: {emotions_response.status_code}')
            
            if emotions_response.status_code == 200:
                emotions_data = emotions_response.json()
                print('✅ Child Emotions berhasil')
                print(f'  - Labels: {emotions_data.get("labels", [])}')
                print(f'  - Values: {emotions_data.get("values", [])}')
            else:
                print(f'❌ Child Emotions gagal: {emotions_response.text[:200]}')
                
        else:
            print(f'❌ Login gagal: {response.text}')
            
    except Exception as e:
        print(f'❌ Error: {e}')

if __name__ == '__main__':
    test_child_details()
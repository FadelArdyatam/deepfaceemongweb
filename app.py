from flask import Flask, render_template, Response, jsonify, request
import cv2
import os
import time
import csv
import shutil
import requests
import base64
import numpy as np
from collections import deque, Counter
from deepface import DeepFace

app = Flask(__name__)

# Configuration for API URL (can be ngrok or localhost)
API_BASE_URL = os.environ.get('API_BASE_URL', 'http://localhost:5000')

# Debug: Print configuration on startup
print(f"🔧 API_BASE_URL configured as: {API_BASE_URL}")
if API_BASE_URL != 'http://localhost:5000':
    print(f"✅ Using ngrok API: {API_BASE_URL}")
else:
    print("⚠️  Using local processing (localhost)")

# Paths and configuration for periodic snapshots
BASE_DIR = os.path.dirname(__file__)
UPLOADS_DIR = os.path.join(BASE_DIR, 'uploads')
LOG_CSV_PATH = os.path.join(UPLOADS_DIR, 'log.csv')
GALLERY_DIR = os.path.join(BASE_DIR, 'gallery')

# Ensure uploads directory exists at startup
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(GALLERY_DIR, exist_ok=True)

# Session tracking (per-process simple approach)
SESSION_START_TS = None

def _append_csv_log(timestamp_iso, dominant_emotion, file_path_relative, identity_label):
    """Append a single log row to CSV, creating header if file does not exist.
    Header columns: timestamp,emotion,file_path,identity
    """
    file_exists = os.path.exists(LOG_CSV_PATH)
    with open(LOG_CSV_PATH, mode='a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(["timestamp", "emotion", "file_path", "identity"])  # header
        writer.writerow([timestamp_iso, dominant_emotion, file_path_relative, identity_label])

def _send_frame_to_ngrok_api(frame, api_url):
    """Send frame to ngrok API for emotion analysis with optimization"""
    try:
        # Clean and validate URL
        api_url = api_url.strip().rstrip('/')
        if not api_url.startswith(('http://', 'https://')):
            print(f"❌ Format URL API tidak valid: {api_url}")
            return None
        
        # Resize frame untuk mengurangi ukuran data
        height, width = frame.shape[:2]
        if width > 640:  # Resize jika terlalu besar
            scale = 640 / width
            new_width = 640
            new_height = int(height * scale)
            frame_resized = cv2.resize(frame, (new_width, new_height))
        else:
            frame_resized = frame
        
        # Encode frame as base64 dengan kompresi yang lebih baik
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]  # Kompresi 70%
        _, buffer = cv2.imencode('.jpg', frame_resized, encode_param)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Send to ngrok API dengan timeout yang lebih pendek
        response = requests.post(
            f"{api_url}/analyze_emotion",
            json={'image': frame_base64},
            timeout=5,  # Kurangi timeout
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ API error: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.ConnectionError as e:
        print(f"❌ Koneksi error ke ngrok API: {e}")
        return None
    except requests.exceptions.Timeout as e:
        print(f"⏰ Timeout error ke ngrok API: {e}")
        return None
    except Exception as e:
        print(f"❌ Error mengirim ke ngrok API: {e}")
        return None


# Camera functions removed - now using client-side camera


# Video stream function removed - now using client-side camera

@app.route('/')
def index():
    global SESSION_START_TS
    SESSION_START_TS = time.time()
    return render_template('index.html')

# Video feed endpoint removed - now using client-side camera

@app.route('/camera/health')
def camera_health():
    # Camera health check no longer needed with client-side camera
    return jsonify({'camera': 'client-side', 'status': 'ok'}), 200

@app.route('/config')
def get_config():
    """Return configuration including API base URL"""
    is_ngrok = API_BASE_URL != 'http://localhost:5000'
    return jsonify({
        'apiBaseUrl': API_BASE_URL,
        'isNgrok': is_ngrok,
        'cameraAvailable': True,  # Client-side camera is always available
        'status': 'ngrok' if is_ngrok else 'local',
        'cameraType': 'client-side'
    })

@app.route('/analyze_emotion', methods=['POST'])
def analyze_emotion():
    """Endpoint to analyze emotion from base64 image"""
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
        
        # Decode base64 image
        image_data = base64.b64decode(data['image'])
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'error': 'Invalid image data'}), 400
        
        # Analyze emotion
        result = DeepFace.analyze(
            frame,
            actions=['emotion'],
            detector_backend='opencv',
            enforce_detection=False,
            silent=True
        )
        
        emotion = result[0]['dominant_emotion']
        
        # Optional: Save frame and log emotion (for analytics)
        if SESSION_START_TS is not None:
            try:
                now = time.time()
                timestamp_iso = time.strftime('%Y-%m-%dT%H-%M-%S', time.localtime(now))
                filename = f"{timestamp_iso}_{emotion}.jpg"
                file_path = os.path.join(UPLOADS_DIR, filename)
                
                # Save frame
                cv2.imwrite(file_path, frame)
                relative_path = os.path.join('uploads', filename)
                
                # Log to CSV
                _append_csv_log(timestamp_iso, emotion, relative_path, "Client-Side")
            except Exception as save_err:
                print(f"Error saving client-side frame: {save_err}")
        
        return jsonify({'emotion': emotion})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/analyze_emotion_faces', methods=['POST'])
def analyze_emotion_faces():
    """Endpoint for emotion analysis with face recognition"""
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
        
        # Decode base64 image
        image_data = base64.b64decode(data['image'])
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'error': 'Invalid image data'}), 400
        
        try:
            # Analyze emotion
            result = DeepFace.analyze(
                frame,
                actions=['emotion'],
                detector_backend='opencv',
                enforce_detection=False,
                silent=True
            )
            
            emotion = result[0]['dominant_emotion']
            
            # Try face recognition against known faces
            recognized_person = "Unknown"
            try:
                # Use DeepFace.find for face recognition
                recognition_results = DeepFace.find(
                    frame,
                    db_path=GALLERY_DIR,
                    model_name='ArcFace',
                    detector_backend='opencv',
                    distance_metric='cosine',
                    enforce_detection=False,
                    silent=True
                )
                
                if recognition_results and len(recognition_results) > 0:
                    df = recognition_results[0] if isinstance(recognition_results, list) else recognition_results
                    if df is not None and hasattr(df, 'empty') and not df.empty:
                        top_result = df.iloc[0]
                        identity_path = str(top_result.get('identity', ''))
                        distance = top_result.get('distance', None)
                        
                        if identity_path and distance is not None and distance <= 0.5:  # Threshold for recognition
                            try:
                                recognized_person = os.path.basename(os.path.dirname(identity_path))
                            except Exception:
                                recognized_person = os.path.splitext(os.path.basename(identity_path))[0]
            except Exception as e:
                print(f"Face recognition error: {e}")
                recognized_person = "Unknown"
            
            # Create face data with recognition info
            h, w = frame.shape[:2]
            face_data = [{
                'x': 0,
                'y': 0,
                'width': int(w),
                'height': int(h),
                'emotion': emotion,
                'confidence': 0.8,
                'person': recognized_person
            }]
            
            # Save frame and log data
            if SESSION_START_TS is not None:
                try:
                    now = time.time()
                    timestamp_iso = time.strftime('%Y-%m-%dT%H-%M-%S', time.localtime(now))
                    filename = f"{timestamp_iso}_{emotion}_{recognized_person}.jpg"
                    file_path = os.path.join(UPLOADS_DIR, filename)
                    
                    # Draw bounding box and labels on frame for saving
                    frame_with_box = frame.copy()
                    cv2.rectangle(frame_with_box, (0, 0), (w, h), (0, 255, 0), 2)
                    cv2.putText(frame_with_box, f"Emotion: {emotion}", (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame_with_box, f"Person: {recognized_person}", (10, 70), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                    
                    cv2.imwrite(file_path, frame_with_box)
                    relative_path = os.path.join('uploads', filename)
                    
                    # Log to CSV
                    _append_csv_log(timestamp_iso, emotion, relative_path, recognized_person)
                except Exception as save_err:
                    print(f"Error saving frame: {save_err}")
            
            return jsonify({
                'overall_emotion': emotion,
                'faces': face_data,
                'face_count': 1,
                'recognized_person': recognized_person
            })
            
        except Exception as e:
            print(f"Error in emotion analysis: {e}")
            # Return error response
            return jsonify({
                'overall_emotion': 'error',
                'faces': [],
                'face_count': 0,
                'recognized_person': 'Unknown',
                'error': str(e)
            })
        
    except Exception as e:
        print(f"Error in analyze_emotion_faces: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/analytics/data')
def analytics_data():
    """Return emotion distribution and timeline since SESSION_START_TS.
    Optional query param: identity=<name> to filter by recognized identity.
    Returns:
      - sessionStart
      - counts: {emotion: count}
      - timeline: [{t, emotion, identity}]
      - identities: [list of identities seen since session]
      - perIdentityCounts: {identity: {emotion: count}}
    """
    try:
        if SESSION_START_TS is None:
            # If session not started yet, return empty
            return jsonify({
                'sessionStart': None,
                'counts': {},
                'timeline': [],
                'identities': [],
                'perIdentityCounts': {}
            })

        # Read CSV and aggregate
        counts = {}
        timeline = []
        identities_set = set()
        per_identity_counts = {}
        session_start_struct = time.localtime(SESSION_START_TS)
        session_start_str = time.strftime('%Y-%m-%dT%H-%M-%S', session_start_struct)
        filter_identity = request.args.get('identity', default=None)
        if filter_identity is not None and filter_identity.strip().lower() in ('all', ''):
            filter_identity = None

        if os.path.exists(LOG_CSV_PATH):
            with open(LOG_CSV_PATH, 'r', encoding='utf-8') as f:
                # Skip header if present
                header = f.readline().strip().split(',')
                # If header is not the expected one, treat it as data
                expected3 = ["timestamp", "emotion", "file_path"]
                expected4 = ["timestamp", "emotion", "file_path", "identity"]
                if header != expected3 and header != expected4:
                    # Process the first line as data
                    try:
                        # Support 3 or 4 columns
                        parts = ','.join(header).split(',')
                        ts_str = parts[0]
                        emotion = parts[1] if len(parts) > 1 else None
                        identity = parts[3] if len(parts) > 3 else ''
                    except ValueError:
                        ts_str, emotion, identity = None, None, ''
                    if ts_str and emotion:
                        try:
                            if ts_str >= session_start_str:
                                if not filter_identity or identity == filter_identity:
                                    counts[emotion] = counts.get(emotion, 0) + 1
                                    timeline.append({'t': ts_str, 'emotion': emotion, 'identity': identity})
                                if identity:
                                    identities_set.add(identity)
                                    d = per_identity_counts.setdefault(identity, {})
                                    d[emotion] = d.get(emotion, 0) + 1
                        except Exception:
                            pass

                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) < 2:
                        continue
                    ts_str, emotion = parts[0], parts[1]
                    identity = parts[3] if len(parts) > 3 else ''
                    # Compare string timestamps lexicographically (safe with fixed format)
                    if ts_str >= session_start_str:
                        if not filter_identity or identity == filter_identity:
                            counts[emotion] = counts.get(emotion, 0) + 1
                            timeline.append({'t': ts_str, 'emotion': emotion, 'identity': identity})
                        if identity:
                            identities_set.add(identity)
                            d = per_identity_counts.setdefault(identity, {})
                            d[emotion] = d.get(emotion, 0) + 1

        return jsonify({
            'sessionStart': session_start_str,
            'counts': counts,
            'timeline': timeline,
            'identities': sorted(list(identities_set)),
            'perIdentityCounts': per_identity_counts
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ---------------- Gallery Management Endpoints ---------------- #

@app.route('/gallery/upload', methods=['POST'])
def gallery_upload():
    """Upload an example image for an identity. Form fields: name, image(file)."""
    try:
        person_name = request.form.get('name', '').strip()
        file = request.files.get('image')
        if not person_name or file is None:
            return jsonify({'error': 'name and image are required'}), 400
        person_dir = os.path.join(GALLERY_DIR, person_name)
        os.makedirs(person_dir, exist_ok=True)
        ts = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        ext = os.path.splitext(file.filename or '')[1].lower() or '.jpg'
        save_path = os.path.join(person_dir, f'{ts}{ext}')
        file.save(save_path)
        return jsonify({'ok': True, 'saved': os.path.relpath(save_path, BASE_DIR)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/gallery/identity/<name>', methods=['DELETE'])
def gallery_delete_identity(name):
    try:
        person_dir = os.path.join(GALLERY_DIR, name)
        if not os.path.exists(person_dir):
            return jsonify({'error': 'identity not found'}), 404
        shutil.rmtree(person_dir)
        return jsonify({'ok': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/gallery/rebuild', methods=['POST'])
def gallery_rebuild():
    """Force rebuild of DeepFace representations by removing cached pkl files."""
    try:
        # Remove cached representation files to trigger rebuild on next find
        for fname in os.listdir(GALLERY_DIR):
            if fname.lower().startswith('representations_') and fname.lower().endswith('.pkl'):
                try:
                    os.remove(os.path.join(GALLERY_DIR, fname))
                except Exception:
                    pass
        return jsonify({'ok': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

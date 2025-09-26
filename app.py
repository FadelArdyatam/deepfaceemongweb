from flask import Flask, render_template, Response, jsonify, request, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_jwt_extended import JWTManager, jwt_required, get_jwt_identity
import cv2
import os
import time
import csv
import shutil
import requests
import base64
import numpy as np
import logging
from collections import deque, Counter
from deepface import DeepFace
from config import Config
from models import db, User, Student, EmotionSession, EmotionLog, StudentTeacher, StudentParent
from auth import auth_bp, require_role
from time import time as now_time

# Simple in-memory throttle cache: {(session_id, student_id): last_ts}
LOG_THROTTLE_CACHE = {}
LOG_THROTTLE_SECONDS = 1.0

# Optional Redis integration
REDIS_URL = os.environ.get('REDIS_URL', '').strip()
redis_client = None
if REDIS_URL:
    try:
        import redis
        redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
        # simple ping
        redis_client.ping()
        print("✅ Redis connected")
        
        # Start background flush job
        try:
            from redis_flush_job import start_flush_job_background
            start_flush_job_background()
        except Exception as flush_err:
            print(f"⚠️  Redis flush job failed to start: {flush_err}")
    except Exception as e:
        print(f"⚠️  Redis unavailable: {e}")
        redis_client = None

def _should_log(session_id: int, student_id: int) -> bool:
    """Throttle decision using Redis if available, else in-memory."""
    try:
        if redis_client is not None and session_id is not None and student_id is not None:
            key = f"emlog:last:{session_id}:{student_id}"
            last = redis_client.get(key)
            now_s = now_time()
            if last is not None:
                try:
                    if (now_s - float(last)) < LOG_THROTTLE_SECONDS:
                        return False
                except Exception:
                    pass
            # store current ts with small TTL safeguard
            redis_client.set(key, str(now_s), ex=5)
            return True
    except Exception:
        pass
    # Fallback in-memory
    key = (session_id, student_id)
    last_ts = LOG_THROTTLE_CACHE.get(key, 0)
    tnow = now_time()
    if (tnow - last_ts) < LOG_THROTTLE_SECONDS:
        return False
    LOG_THROTTLE_CACHE[key] = tnow
    return True

def _agg_increment_today(teacher_id: int, emotion: str) -> None:
    """Optional lightweight aggregation in Redis for today per teacher."""
    if not redis_client or not teacher_id or not emotion:
        return
    try:
        today_str = datetime.utcnow().strftime('%Y-%m-%d')
        key = f"emagg:{teacher_id}:{today_str}"
        redis_client.hincrby(key, emotion, 1)
        redis_client.expire(key, 3 * 24 * 3600)  # keep few days
    except Exception:
        pass
from datetime import datetime

app = Flask(__name__)
app.config.from_object(Config)

# Initialize extensions
db.init_app(app)
migrate = Migrate(app, db)
jwt = JWTManager(app)

# Register blueprints
app.register_blueprint(auth_bp, url_prefix='/api/auth')

# Configuration for API URL (can be ngrok or localhost)
API_BASE_URL = os.environ.get('API_BASE_URL', 'http://localhost:5000')
CAM_SOURCE = os.environ.get('CAM_SOURCE', 'webcam').lower()  # 'webcam' | 'rtsp'
RTSP_URL_ENV = os.environ.get('RTSP_URL', '').strip()

# Runtime-overridable camera source
CURRENT_CAM_SOURCE = CAM_SOURCE
CURRENT_RTSP_URL = RTSP_URL_ENV

# Runtime-overridable detector backend
CURRENT_DETECTOR_BACKEND = 'opencv'  # opencv, retinaface, mtcnn

# Debug: Print configuration on startup
print(f"🔧 API_BASE_URL configured as: {API_BASE_URL}")
if API_BASE_URL != 'http://localhost:5000':
    print(f"✅ Using ngrok API: {API_BASE_URL}")
else:
    print("⚠️  Using local processing (localhost)")

# Setup basic logging
if not app.logger.handlers:
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s in %(module)s: %(message)s')
app.logger.setLevel(logging.INFO)

# Paths and configuration for periodic snapshots
BASE_DIR = os.path.dirname(__file__)
UPLOADS_DIR = os.path.join(BASE_DIR, 'uploads')
LOG_CSV_PATH = os.path.join(UPLOADS_DIR, 'log.csv')
GALLERY_DIR = os.path.join(BASE_DIR, 'gallery')  # legacy (tidak dipakai lagi untuk pencocokan)
KNOWN_FACES_DIR = os.path.join(BASE_DIR, 'known_faces')

# Ensure uploads directory exists at startup
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(GALLERY_DIR, exist_ok=True)
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

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


def _open_camera_with_fallback():
    """Try opening camera with multiple indices and backends (Windows friendly)."""
    # Prioritize DirectShow for Windows (berdasarkan test), then others
    preferred_backends = [ cv2.CAP_MSMF,cv2.CAP_DSHOW, cv2.CAP_ANY]
    indices_to_try = [0, 1, 2]
    
    print("🔍 Mencoba membuka camera...")
    
    # Try with specific backends first
    for backend in preferred_backends:
        backend_name = {
            cv2.CAP_DSHOW: "DirectShow",
            cv2.CAP_MSMF: "Media Foundation", 
            cv2.CAP_ANY: "Any Available"
        }.get(backend, f"Backend {backend}")
        
        for idx in indices_to_try:
            try:
                print(f"  Mencoba {backend_name} pada index {idx}...")
                cap = cv2.VideoCapture(idx, backend)
                
                if cap.isOpened():
                    # Test if we can actually read a frame dengan retry
                    retry_count = 0
                    max_retries = 3
                    success = False
                    
                    while retry_count < max_retries and not success:
                        ret, frame = cap.read()
                        if ret and frame is not None and frame.size > 0:
                            success = True
                        else:
                            retry_count += 1
                            time.sleep(0.1)  # Tunggu sebentar sebelum retry
                    
                    if success:
                        print(f"✅ Camera berhasil dibuka dengan {backend_name} pada index {idx}")
                        # Set camera properties for better performance
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        cap.set(cv2.CAP_PROP_FPS, 30)
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer size
                        return cap
                    else:
                        print(f"  ❌ Tidak bisa membaca frame dari {backend_name} index {idx} setelah {max_retries} percobaan")
                cap.release()
            except Exception as e:
                print(f"  ❌ Error dengan {backend_name} index {idx}: {e}")
                pass
    
    # Fallback: try default constructor without backend
    print("  Mencoba default backend...")
    try:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            # Test dengan retry juga
            retry_count = 0
            max_retries = 3
            success = False
            
            while retry_count < max_retries and not success:
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    success = True
                else:
                    retry_count += 1
                    time.sleep(0.1)
            
            if success:
                print("✅ Camera berhasil dibuka dengan default backend")
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                return cap
        cap.release()
    except Exception as e:
        print(f"❌ Error dengan default backend: {e}")
    
    print("⚠️  Peringatan: Tidak ada camera yang bisa dibuka")
    return None


def _open_rtsp_stream(rtsp_url: str):
    """Open RTSP stream with retries."""
    try:
        if not rtsp_url or not (rtsp_url.startswith('rtsp://') or rtsp_url.startswith('rtmp://')):
            print("❌ RTSP URL tidak valid")
            return None
        print(f"🔗 Membuka RTSP: {rtsp_url}")
        # Prefer FFMPEG backend if available
        cap = cv2.VideoCapture(rtsp_url)
        retries = 0
        while retries < 10:
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    print("✅ RTSP stream terbuka")
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    return cap
            retries += 1
            time.sleep(0.3)
        try:
            cap.release()
        except Exception:
            pass
        print("❌ Gagal membuka RTSP stream")
        return None
    except Exception as e:
        print(f"❌ RTSP error: {e}")
        return None


def _open_video_source():
    """Open video source based on CURRENT_CAM_SOURCE."""
    global CURRENT_CAM_SOURCE, CURRENT_RTSP_URL
    if CURRENT_CAM_SOURCE == 'rtsp':
        return _open_rtsp_stream(CURRENT_RTSP_URL)
    return _open_camera_with_fallback()


def generate_frames():
    cap = _open_video_source()
    if cap is None or not cap.isOpened():
        print("❌ Error: Tidak bisa mengakses camera")
        # Return error frame instead of breaking
        error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(error_frame, "Camera tidak tersedia", (50, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        ret, buffer = cv2.imencode('.jpg', error_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        return

    last_saved_ts = 0.0
    save_interval_seconds = 5.0
    frame_count = 0
    recognition_interval_frames = 10  # Kurangi frekuensi untuk performa
    recognized_label = "Unknown"
    recognized_distance = None
    emotion_history: deque[str] = deque(maxlen=5)
    consecutive_failures = 0
    max_consecutive_failures = 20  # Tingkatkan tolerance
    frame_skip_count = 0
    max_frame_skips = 5

    print("🎥 Memulai video stream...")
    
    # Tunggu sebentar untuk camera stabil
    print("⏳ Menunggu camera stabil...")
    time.sleep(2)
    
    while True:
        try:
            success, frame = cap.read()  # Read a frame from the webcam
            if not success:
                consecutive_failures += 1
                frame_skip_count += 1
                
                if frame_skip_count <= max_frame_skips:
                    print(f"⏭️  Skip frame {frame_skip_count}/{max_frame_skips}")
                    time.sleep(0.05)  # Tunggu lebih singkat
                    continue
                else:
                    print(f"⚠️  Gagal membaca frame ({consecutive_failures}/{max_consecutive_failures})")
                    if consecutive_failures >= max_consecutive_failures:
                        print("❌ Terlalu banyak kegagalan, menghentikan stream")
                        break
                    time.sleep(0.1)  # Tunggu sebentar sebelum coba lagi
                    frame_skip_count = 0  # Reset skip counter
                    continue
            
            # Reset counters jika berhasil
            consecutive_failures = 0
            frame_skip_count = 0
            
        except Exception as e:
            print(f"❌ Error membaca frame: {e}")
            consecutive_failures += 1
            if consecutive_failures >= max_consecutive_failures:
                break
            continue
        
        try:
            # Try to use ngrok API first if configured (hanya setiap beberapa frame)
            if API_BASE_URL != 'http://localhost:5000' and frame_count % 5 == 0:
                api_result = _send_frame_to_ngrok_api(frame, API_BASE_URL)
                if api_result and 'emotion' in api_result:
                    emotion = api_result['emotion']
                    print(f"🎯 Emotion dari Colab: {emotion}")
                else:
                    # Fallback to local DeepFace jika ngrok gagal
                    print("🔄 Fallback ke local processing...")
                    result = DeepFace.analyze(
                        frame,
                        actions=['emotion'],
                        detector_backend='opencv',
                        enforce_detection=False,
                        silent=True
                    )
                    emotion = result[0]['dominant_emotion']
            else:
                # Use local DeepFace untuk frame lainnya
                if frame_count % 10 == 0:  # Hanya proses setiap 10 frame untuk performa
                    result = DeepFace.analyze(
                        frame,
                        actions=['emotion'],
                        detector_backend='opencv',
                        enforce_detection=False,
                        silent=True
                    )
                    emotion = result[0]['dominant_emotion']
                else:
                    # Gunakan emotion terakhir jika tidak memproses frame ini
                    emotion = emotion_history[-1] if emotion_history else "unknown"
            
            emotion_history.append(emotion)
            # Smoothing using mode of recent emotions
            if len(emotion_history) > 0:
                emotion = Counter(emotion_history).most_common(1)[0][0]
            
            # Add emotion text overlay to the frame
            cv2.putText(frame, f'Emotion: {emotion}', (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Tampilkan status processing
            status_text = "Colab" if API_BASE_URL != 'http://localhost:5000' and frame_count % 5 == 0 else "Local"
            cv2.putText(frame, f'Mode: {status_text}', (30, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                        
        except Exception as e:
            print(f"❌ Error dengan emotion detection: {e}")
            emotion = "unknown"
            cv2.putText(frame, f'Error: {str(e)[:30]}...', (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Periodically run face recognition (1:N) against gallery
        try:
            if frame_count % recognition_interval_frames == 0:
                # Detect faces and use cropped ROI(s) for matching to handle small/distant faces
                detections = DeepFace.extract_faces(
                    img_path=frame,
                    detector_backend='opencv',
                    align=True,
                    enforce_detection=False
                )
                roi_list = []
                for det in detections:
                    face_img = det.get('face')
                    if face_img is None:
                        continue
                    # face_img is RGB float [0..1]; convert to BGR uint8
                    face_bgr = (face_img[:, :, ::-1] * 255).astype('uint8')
                    roi_list.append(face_bgr)

                if not roi_list:
                    roi_list = [frame]

                best_name = "Unknown"
                best_distance = None
                threshold = 0.5
                for roi in roi_list:
                    results = DeepFace.find(
                        roi,
                        db_path=KNOWN_FACES_DIR,
                        model_name='ArcFace',
                        detector_backend='opencv',
                        distance_metric='cosine',
                        enforce_detection=False,
                        silent=True
                    )
                    df = results[0] if isinstance(results, list) else results
                    if df is not None and hasattr(df, 'empty') and not df.empty:
                        top = df.iloc[0]
                        identity_path = str(top.get('identity', ''))
                        distance = top.get('distance', None)
                        if identity_path:
                            try:
                                person_name = os.path.basename(os.path.dirname(identity_path))
                            except Exception:
                                person_name = os.path.splitext(os.path.basename(identity_path))[0]
                            if distance is None or distance <= threshold:
                                if best_distance is None or (distance is not None and distance < best_distance):
                                    best_name = person_name
                                    best_distance = distance
                if best_distance is not None:
                    recognized_label = best_name
                    recognized_distance = best_distance
                else:
                    recognized_label = "Unknown"
                    recognized_distance = None
        except Exception as e:
            print("Error with face recognition:", e)

        # Overlay recognized name + distance
        id_text = f'ID: {recognized_label}' + (f' ({recognized_distance:.2f})' if recognized_distance is not None else '')
        cv2.putText(frame, id_text, (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        # Periodic snapshot capture and logging
        now = time.time()
        if now - last_saved_ts >= save_interval_seconds:
            last_saved_ts = now
            timestamp_iso = time.strftime('%Y-%m-%dT%H-%M-%S', time.localtime(now))
            filename = f"{timestamp_iso}_{emotion}.jpg"
            file_path = os.path.join(UPLOADS_DIR, filename)

            try:
                # Save current frame as JPEG
                cv2.imwrite(file_path, frame)
                # Store relative path for portability
                relative_path = os.path.join('uploads', filename)
                _append_csv_log(timestamp_iso, emotion, relative_path, recognized_label)
            except Exception as save_err:
                print("Error saving snapshot or logging:", save_err)

        # Convert the frame to JPEG format for the web stream
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        frame_count += 1

        # Yield the frame as part of the MJPEG stream
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    # Cleanup ketika loop berakhir
    print("🧹 Membersihkan resources...")
    if cap:
        cap.release()
    print("✅ Video stream berakhir")

@app.route('/')
def index():
    """Main page - redirect to login"""
    return redirect(url_for('login'))

@app.route('/emotion-detection')
def emotion_detection():
    """Original emotion detection page"""
    global SESSION_START_TS
    SESSION_START_TS = time.time()
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/camera/health')
def camera_health():
    cap = _open_video_source()
    ok = bool(cap and cap.isOpened())
    if ok:
        cap.release()
    return jsonify({'camera': 'ok' if ok else 'unavailable'}), (200 if ok else 503)

@app.route('/config')
def get_config():
    """Return configuration including API base URL"""
    is_ngrok = API_BASE_URL != 'http://localhost:5000'
    return jsonify({
        'apiBaseUrl': API_BASE_URL,
        'isNgrok': is_ngrok,
        'cameraAvailable': True,
        'status': 'ngrok' if is_ngrok else 'local',
        'cameraSource': CURRENT_CAM_SOURCE,
        'rtspUrl': (CURRENT_RTSP_URL[:15] + '...' if CURRENT_RTSP_URL else '')
    })

@app.route('/camera/source', methods=['POST'])
def set_camera_source():
    """Set camera source at runtime. Body: {source:'webcam'|'rtsp', rtspUrl?:string}"""
    global CURRENT_CAM_SOURCE, CURRENT_RTSP_URL
    try:
        data = request.get_json() or {}
        src = str(data.get('source', CURRENT_CAM_SOURCE)).lower()
        if src not in ('webcam', 'rtsp'):
            return jsonify({'error': 'source harus webcam atau rtsp'}), 400
        if src == 'rtsp':
            url = data.get('rtspUrl', CURRENT_RTSP_URL)
            if not url:
                return jsonify({'error': 'rtspUrl harus diisi untuk source=rtsp'}), 400
            CURRENT_RTSP_URL = url
        CURRENT_CAM_SOURCE = src
        return jsonify({'message': 'Camera source updated', 'cameraSource': CURRENT_CAM_SOURCE}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/detector/backend', methods=['GET'])
def get_detector_backend():
    """Get current detector backend"""
    return jsonify({'detectorBackend': CURRENT_DETECTOR_BACKEND}), 200

@app.route('/detector/backend', methods=['POST'])
def set_detector_backend():
    """Set detector backend at runtime. Body: {backend:'opencv'|'retinaface'|'mtcnn'}"""
    global CURRENT_DETECTOR_BACKEND
    try:
        data = request.get_json() or {}
        backend = str(data.get('backend', CURRENT_DETECTOR_BACKEND)).lower()
        if backend not in ('opencv', 'retinaface', 'mtcnn'):
            return jsonify({'error': 'backend harus opencv, retinaface, atau mtcnn'}), 400
        CURRENT_DETECTOR_BACKEND = backend
        print(f"🔍 Detector backend changed to: {backend}")
        return jsonify({'message': 'Detector backend updated', 'detectorBackend': CURRENT_DETECTOR_BACKEND}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

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
        
        # Optional attribution info
        provided_session_id = data.get('session_id')
        provided_teacher_id = data.get('teacher_id')
        if isinstance(provided_teacher_id, str) and provided_teacher_id.isdigit():
            provided_teacher_id = int(provided_teacher_id)
        if isinstance(provided_session_id, str) and str(provided_session_id).isdigit():
            provided_session_id = int(provided_session_id)

        # Detect faces and analyze emotion per face + identify student by known_faces folder name
        detections = []
        try:
            faces = DeepFace.extract_faces(
                img_path=frame,
                detector_backend=CURRENT_DETECTOR_BACKEND,
                align=True,
                enforce_detection=False
            )
        except Exception:
            faces = []

        if faces:
            for det in faces:
                face_img = det.get('face')
                region = det.get('facial_area') or det.get('region') or {}
                # region may contain keys x, y, w, h
                if face_img is None:
                    continue
                # Convert aligned face back to BGR uint8 for analysis
                face_bgr = (face_img[:, :, ::-1] * 255).astype('uint8')
                analysis = DeepFace.analyze(
                    face_bgr,
            actions=['emotion'],
            detector_backend='opencv',
            enforce_detection=False,
            silent=True
        )
                emo = None
                try:
                    emo = analysis[0]['dominant_emotion']
                except Exception:
                    pass

                # Identify via known_faces
                identity = None
                distance = None
                try:
                    search = DeepFace.find(
                        face_bgr,
                        db_path=KNOWN_FACES_DIR,
                        model_name='ArcFace',
                        detector_backend='skip',  # Skip detection since face is already extracted
                        distance_metric='cosine',
                        enforce_detection=False,
                        silent=True
                    )
                    df = search[0] if isinstance(search, list) else search
                    if df is not None and hasattr(df, 'empty') and not df.empty:
                        top = df.iloc[0]
                        identity_path = str(top.get('identity', ''))
                        distance = float(top.get('distance', None)) if top.get('distance', None) is not None else None
                        if identity_path:
                            try:
                                identity = os.path.basename(os.path.dirname(identity_path))
                            except Exception:
                                identity = os.path.splitext(os.path.basename(identity_path))[0]
                except Exception as _:
                    pass

                det_entry = {
                    'x': int(region.get('x', 0)),
                    'y': int(region.get('y', 0)),
                    'w': int(region.get('w', face_bgr.shape[1] if face_bgr is not None else 0)),
                    'h': int(region.get('h', face_bgr.shape[0] if face_bgr is not None else 0)),
                    'emotion': emo,
                    'identity': identity,
                    'distance': distance
                }
                detections.append(det_entry)

            # Dominant emotion overall: mode of per-face emotions (if any)
            emos = [d['emotion'] for d in detections if d.get('emotion')]
            overall = Counter(emos).most_common(1)[0][0] if emos else None

            # Auto-log to DB per detected student (if matched)
            try:
                # Resolve session target
                target_session = None
                if provided_session_id:
                    target_session = EmotionSession.query.get(provided_session_id)
                elif provided_teacher_id:
                    target_session = EmotionSession.query.filter_by(teacher_id=provided_teacher_id, status='active').order_by(EmotionSession.start_time.desc()).first()
                    if not target_session:
                        target_session = EmotionSession(student_id=None, teacher_id=provided_teacher_id, session_name='Live Guru', status='active')
                        db.session.add(target_session)
                        db.session.commit()

                for d in detections:
                    if not d.get('identity') or not d.get('emotion'):
                        continue
                    # Map identity (folder name) to student_code
                    student = Student.query.filter_by(student_code=d['identity']).first()
                    if not student:
                        continue
                    # Choose session: provided target or auto-monitoring per student
                    session_row = target_session
                    if not session_row:
                        session_row = EmotionSession.query.filter_by(student_id=student.id, status='active', teacher_id=None).first()
                        if not session_row:
                            session_row = EmotionSession(student_id=student.id, teacher_id=None, session_name='Auto Monitoring', status='active')
                            db.session.add(session_row)
                            db.session.commit()
                    # Throttle logging per (session_id, student_id)
                    if not _should_log(session_row.id, student.id):
                        continue
                    # Create EmotionLog
                    log = EmotionLog(
                        session_id=session_row.id,
                        student_id=student.id,
                        emotion=d['emotion'],
                        confidence_score=(1.0 - float(d['distance'])) if d.get('distance') is not None else None,
                        image_path=None
                    )
                    db.session.add(log)
                    # Optional aggregation per teacher for today
                    try:
                        tid = provided_teacher_id or session_row.teacher_id
                        if tid:
                            _agg_increment_today(int(tid), d['emotion'])
                    except Exception:
                        pass
                db.session.commit()
            except Exception as _e:
                db.session.rollback()

            return jsonify({'emotion': overall, 'boxes': detections})
        else:
            # No faces detected; analyze full frame once for compatibility
            analysis = DeepFace.analyze(
                frame,
                actions=['emotion'],
                detector_backend='opencv',
                enforce_detection=False,
                silent=True
            )
            emotion = analysis[0]['dominant_emotion']
            return jsonify({'emotion': emotion, 'boxes': []})
        
    except Exception as e:
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

# Dashboard Routes
@app.route('/dashboard')
def dashboard_redirect():
    """Redirect to login if not authenticated, otherwise redirect to appropriate dashboard"""
    if not session.get('user_id'):
        return redirect(url_for('login'))
    
    user = User.query.get(session['user_id'])
    if not user:
        return redirect(url_for('login'))
    
    if user.role == 'guru':
        return redirect(url_for('dashboard_guru'))
    elif user.role == 'orang_tua':
        return redirect(url_for('dashboard_parent'))
    elif user.role == 'admin':
        return redirect(url_for('dashboard_admin'))
    
    return redirect(url_for('login'))

@app.route('/login')
def login():
    """Login page"""
    return render_template('login.html')

@app.route('/dashboard/guru')
def dashboard_guru():
    """Dashboard untuk guru"""
    return render_template('dashboard_guru.html')

@app.route('/dashboard/parent')
def dashboard_parent():
    """Dashboard untuk orang tua"""
    return render_template('dashboard_parent.html')

@app.route('/dashboard/admin')
def dashboard_admin():
    """Dashboard untuk admin"""
    return render_template('dashboard_admin.html')

# API Routes untuk Dashboard
@app.route('/api/dashboard/guru/stats')
@jwt_required()
@require_role(['guru', 'admin'])
def guru_dashboard_stats():
    """API untuk statistik dashboard guru"""
    try:
        user_id = get_jwt_identity()
        user_id = int(user_id) if user_id is not None else None
        
        # Hitung total siswa yang diajar oleh guru ini
        total_students = db.session.query(Student).join(StudentTeacher).filter(
            StudentTeacher.teacher_id == user_id
        ).count()
        
        # Hitung sesi aktif
        active_sessions = EmotionSession.query.filter(
            EmotionSession.teacher_id == user_id,
            EmotionSession.status == 'active'
        ).count()
        
        # Hitung deteksi hari ini
        from datetime import datetime, date
        today = date.today()
        today_detections = db.session.query(EmotionLog).join(EmotionSession).filter(
            EmotionSession.teacher_id == user_id,
            db.func.date(EmotionLog.detected_at) == today
        ).count()
        
        # Data emosi hari ini
        emotion_data = db.session.query(
            EmotionLog.emotion,
            db.func.count(EmotionLog.id).label('count')
        ).join(EmotionSession).filter(
            EmotionSession.teacher_id == user_id,
            db.func.date(EmotionLog.detected_at) == today
        ).group_by(EmotionLog.emotion).all()
        
        emotion_dict = {item.emotion: item.count for item in emotion_data}
        
        return jsonify({
            'total_students': total_students,
            'active_sessions': active_sessions,
            'today_detections': today_detections,
            'avg_emotion': 'Happy' if not emotion_dict else max(emotion_dict, key=emotion_dict.get),
            'emotion_data': emotion_dict
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/dashboard/guru/daily-summary')
@jwt_required()
@require_role(['guru', 'admin'])
def guru_daily_summary():
    """Ringkasan emosi per hari (7 hari terakhir) untuk laporan grafik."""
    try:
        user_id = get_jwt_identity()
        user_id = int(user_id) if user_id is not None else None
        from datetime import date, timedelta
        start = date.today() - timedelta(days=6)
        rows = db.session.query(
            db.func.date(EmotionLog.detected_at).label('d'),
            EmotionLog.emotion,
            db.func.count(EmotionLog.id)
        ).join(EmotionSession).filter(
            EmotionSession.teacher_id == user_id,
            db.func.date(EmotionLog.detected_at) >= start
        ).group_by(db.func.date(EmotionLog.detected_at), EmotionLog.emotion).order_by(db.func.date(EmotionLog.detected_at)).all()
        data = {}
        for d, em, cnt in rows:
            key = str(d)
            if key not in data: data[key] = {}
            data[key][em] = int(cnt)
        return jsonify({ 'start': str(start), 'days': data })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/students')
@jwt_required()
@require_role(['guru', 'admin'])
def get_students():
    """API untuk mendapatkan daftar siswa"""
    try:
        user_id = get_jwt_identity()
        user_id = int(user_id) if user_id is not None else None
        user = User.query.get(user_id)
        
        if user.role == 'admin':
            # Admin bisa lihat semua siswa
            students = Student.query.filter_by(is_active=True).all()
        else:
            # Guru hanya bisa lihat siswa yang dia ajar
            students = db.session.query(Student).join(StudentTeacher).filter(
                StudentTeacher.teacher_id == user_id,
                Student.is_active == True
            ).all()
        
        return jsonify([student.to_dict() for student in students])
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/students', methods=['POST'])
@jwt_required()
@require_role(['guru', 'admin'])
def create_student():
    """API untuk menambahkan siswa baru"""
    try:
        data = None
        if request.is_json:
            data = request.get_json()
        else:
            # Fallback for form-encoded
            data = request.form.to_dict()
        user_id = get_jwt_identity()
        user_id = int(user_id) if user_id is not None else None
        
        # Validasi input
        required_fields = ['student_code', 'full_name', 'class_name']
        for field in required_fields:
            if field not in data or not data[field]:
                return jsonify({'error': f'Field {field} harus diisi'}), 400
        
        # Cek apakah student_code sudah ada
        if Student.query.filter_by(student_code=data['student_code']).first():
            return jsonify({'error': 'Kode siswa sudah digunakan'}), 409
        
        # Buat siswa baru
        birth_date_value = data.get('birth_date')
        if birth_date_value:
            from datetime import datetime as dt
            try:
                # Accept YYYY-MM-DD
                birth_date_value = dt.strptime(birth_date_value[:10], '%Y-%m-%d').date()
            except Exception:
                birth_date_value = None
        student = Student(
            student_code=data['student_code'],
            full_name=data['full_name'],
            class_name=data['class_name'],
            birth_date=birth_date_value,
            photo_path=data.get('photo_path')
        )
        
        db.session.add(student)
        db.session.commit()
        
        # Jika ada photo_path string lokal, coba salin (opsional)
        if data.get('photo_path'):
            try:
                create_known_face_folder(student.student_code, data['photo_path'])
            except Exception:
                pass
        
        # Jika guru yang menambahkan, buat relasi guru-siswa
        user = User.query.get(user_id)
        if user.role == 'guru':
            student_teacher = StudentTeacher(
                student_id=student.id,
                teacher_id=user_id,
                subject=str(data.get('subject', 'Umum'))
            )
            db.session.add(student_teacher)
            db.session.commit()
        
        return jsonify({
            'message': 'Siswa berhasil ditambahkan',
            'student': student.to_dict()
        }), 201
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

def create_known_face_folder(student_code, photo_path):
    """Buat folder untuk known face siswa"""
    try:
        # Buat folder di known_faces
        known_faces_dir = os.path.join(BASE_DIR, 'known_faces')
        student_dir = os.path.join(known_faces_dir, student_code)
        os.makedirs(student_dir, exist_ok=True)
        
        # Copy foto ke folder known_faces
        if os.path.exists(photo_path):
            import shutil
            filename = os.path.basename(photo_path)
            dest_path = os.path.join(student_dir, filename)
            shutil.copy2(photo_path, dest_path)
            print(f"✅ Foto siswa {student_code} berhasil disalin ke known_faces")
        
    except Exception as e:
        print(f"❌ Error membuat known face folder: {e}")

@app.route('/api/students/<int:student_id>', methods=['PUT'])
@jwt_required()
@require_role(['guru', 'admin'])
def update_student(student_id):
    """API untuk update data siswa"""
    try:
        data = request.get_json()
        user_id = get_jwt_identity()
        user_id = int(user_id) if user_id is not None else None
        
        student = Student.query.get(student_id)
        if not student:
            return jsonify({'error': 'Siswa tidak ditemukan'}), 404
        
        # Cek apakah guru berhak mengedit siswa ini
        user = User.query.get(user_id)
        if user.role == 'guru':
            if not db.session.query(StudentTeacher).filter(
                StudentTeacher.teacher_id == user_id,
                StudentTeacher.student_id == student_id
            ).first():
                return jsonify({'error': 'Anda tidak berhak mengedit siswa ini'}), 403
        
        # Update data
        if 'full_name' in data:
            student.full_name = data['full_name']
        if 'class_name' in data:
            student.class_name = data['class_name']
        if 'birth_date' in data:
            student.birth_date = data['birth_date']
        if 'photo_path' in data:
            student.photo_path = data['photo_path']
            # Update known faces jika ada foto baru
            if data['photo_path']:
                create_known_face_folder(student.student_code, data['photo_path'])
        
        student.updated_at = datetime.utcnow()
        db.session.commit()
        
        return jsonify({
            'message': 'Data siswa berhasil diperbarui',
            'student': student.to_dict()
        }), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/api/students/<int:student_id>', methods=['DELETE'])
@jwt_required()
@require_role(['guru', 'admin'])
def delete_student(student_id):
    """API untuk menghapus siswa"""
    try:
        user_id = get_jwt_identity()
        user_id = int(user_id) if user_id is not None else None
        
        student = Student.query.get(student_id)
        if not student:
            return jsonify({'error': 'Siswa tidak ditemukan'}), 404
        
        # Cek apakah guru berhak menghapus siswa ini
        user = User.query.get(user_id)
        if user.role == 'guru':
            if not db.session.query(StudentTeacher).filter(
                StudentTeacher.teacher_id == user_id,
                StudentTeacher.student_id == student_id
            ).first():
                return jsonify({'error': 'Anda tidak berhak menghapus siswa ini'}), 403
        
        # Soft delete
        student.is_active = False
        student.updated_at = datetime.utcnow()
        db.session.commit()
        
        return jsonify({'message': 'Siswa berhasil dihapus'}), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/api/sessions', methods=['POST'])
@jwt_required()
@require_role(['guru', 'admin'])
def create_session():
    """API untuk membuat sesi deteksi emosi baru"""
    try:
        data = request.get_json()
        user_id = get_jwt_identity()
        user_id = int(user_id) if user_id is not None else None
        app.logger.info(f"create_session called by user_id=%s payload=%s", user_id, data)
        
        # Validasi input
        if 'session_name' not in data or 'student_id' not in data:
            return jsonify({'error': 'session_name dan student_id harus diisi'}), 400
        
        # Mode kelas (student_id==0) diperbolehkan: sesi tanpa siswa spesifik
        student_id_val = int(data.get('student_id') or 0)
        if student_id_val > 0:
            student = Student.query.get(student_id_val)
        if not student:
            return jsonify({'error': 'Siswa tidak ditemukan'}), 404
        # Cek apakah guru berhak mengajar siswa ini
        if not db.session.query(StudentTeacher).filter(
            StudentTeacher.teacher_id == user_id,
                StudentTeacher.student_id == student_id_val
        ).first():
            return jsonify({'error': 'Anda tidak berhak mengajar siswa ini'}), 403
        
        # Buat sesi baru
        session = EmotionSession(
            student_id=(student_id_val if student_id_val>0 else None),
            teacher_id=user_id,
            session_name=data['session_name'],
            notes=data.get('notes', ''),
            status='active'
        )
        
        db.session.add(session)
        db.session.commit()
        app.logger.info(f"create_session success id=%s teacher_id=%s student_id=%s", session.id, session.teacher_id, session.student_id)
        
        return jsonify(session.to_dict()), 201
        
    except Exception as e:
        db.session.rollback()
        app.logger.exception("create_session failed")
        return jsonify({'error': str(e)}), 500

@app.route('/api/sessions/<int:session_id>/stop', methods=['POST'])
@jwt_required()
@require_role(['guru', 'admin'])
def stop_session(session_id):
    """API untuk menghentikan sesi deteksi emosi"""
    try:
        user_id = get_jwt_identity()
        user_id = int(user_id) if user_id is not None else None
        app.logger.info("stop_session called by user_id=%s session_id=%s query=%s", user_id, session_id, dict(request.args))
        
        # Optional bulk stop
        stop_all = request.args.get('all', 'false').lower() in ('1', 'true', 'yes')
        user = User.query.get(user_id)
        role = user.role if user else None
        app.logger.info("stop_session role=%s", role)

        stopped_ids = []
        if stop_all:
            # Hentikan semua sesi aktif milik guru ini, dan sesi auto-monitoring (teacher_id NULL)
            q = EmotionSession.query.filter(EmotionSession.status == 'active')
            if role != 'admin':
                q = q.filter((EmotionSession.teacher_id == user_id) | (EmotionSession.teacher_id.is_(None)))
            sessions = q.all()
            app.logger.info("stop_session bulk candidates=%s", [s.id for s in sessions])
            for s in sessions:
                s.status = 'completed'
                s.end_time = datetime.utcnow()
                stopped_ids.append(s.id)
            try:
                db.session.commit()
                app.logger.info("stop_session bulk success count=%s ids=%s", len(stopped_ids), stopped_ids)
                return jsonify({'message': 'Semua sesi aktif dihentikan', 'stopped_ids': stopped_ids}), 200
            except Exception as commit_err:
                db.session.rollback()
                app.logger.exception("stop_session bulk failed")
                return jsonify({'error': f'Gagal bulk stop: {str(commit_err)}'}), 500
        else:
            session = EmotionSession.query.get(session_id)
            if not session:
                app.logger.warning("stop_session session not found id=%s", session_id)
                return jsonify({'error': 'Sesi tidak ditemukan'}), 404
            app.logger.info("stop_session current session teacher_id=%s status=%s", session.teacher_id, session.status)
            try:
                teacher_id_val = int(session.teacher_id) if session.teacher_id is not None else None
            except Exception:
                teacher_id_val = session.teacher_id
            if role != 'admin':
                if teacher_id_val is not None and teacher_id_val != user_id:
                    app.logger.warning("stop_session forbidden user_id=%s teacher_id=%s", user_id, teacher_id_val)
                    return jsonify({'error': 'Anda tidak berhak menghentikan sesi ini'}), 403
            if session.status == 'completed':
                app.logger.info("stop_session already completed id=%s", session.id)
                return jsonify({'message': 'Sesi sudah dihentikan'}), 200
            session.status = 'completed'
            session.end_time = datetime.utcnow()
            app.logger.info("stop_session updating id=%s -> completed", session.id)
            try:
                db.session.commit()
                app.logger.info("stop_session success id=%s", session.id)
                return jsonify({'message': 'Sesi berhasil dihentikan', 'session_id': session.id}), 200
            except Exception as commit_err:
                db.session.rollback()
                app.logger.exception("stop_session failed commit")
                return jsonify({'error': f'Gagal menyimpan perubahan: {str(commit_err)}'}), 500
        
    except Exception as e:
        db.session.rollback()
        app.logger.exception("stop_session unexpected error")
        return jsonify({'error': f'Unexpected: {str(e)}'}), 500

@app.route('/api/sessions/active')
@jwt_required()
@require_role(['guru', 'admin'])
def list_active_sessions():
    """Endpoint debug: daftar sesi aktif milik guru ini (admin melihat semua)."""
    try:
        user_id = get_jwt_identity()
        user_id = int(user_id) if user_id is not None else None
        user = User.query.get(user_id)
        role = user.role if user else None
        q = EmotionSession.query.filter(EmotionSession.status == 'active')
        if role != 'admin':
            q = q.filter((EmotionSession.teacher_id == user_id) | (EmotionSession.teacher_id.is_(None)))
        sessions = q.order_by(EmotionSession.start_time.desc()).all()
        return jsonify([s.to_dict() for s in sessions]), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/sessions/<int:session_id>', methods=['GET'])
@jwt_required()
@require_role(['guru', 'admin'])
def get_session(session_id):
    try:
        s = EmotionSession.query.get(session_id)
        if not s:
            return jsonify({'error': 'Sesi tidak ditemukan'}), 404
        return jsonify(s.to_dict()), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# API Routes untuk Dashboard Orang Tua
@app.route('/api/dashboard/parent/stats')
@jwt_required()
@require_role(['orang_tua'])
def parent_dashboard_stats():
    """API untuk statistik dashboard orang tua"""
    try:
        user_id = get_jwt_identity()
        user_id = int(user_id) if user_id is not None else None
        
        # Hitung total anak
        total_children = db.session.query(Student).join(StudentParent).filter(
            StudentParent.parent_id == user_id,
            Student.is_active == True
        ).count()
        
        # Hitung sesi minggu ini
        from datetime import datetime, date, timedelta
        week_ago = date.today() - timedelta(days=7)
        weekly_sessions = db.session.query(EmotionSession).join(Student).join(StudentParent).filter(
            StudentParent.parent_id == user_id,
            db.func.date(EmotionSession.start_time) >= week_ago
        ).count()
        
        # Data emosi minggu ini
        emotion_data = db.session.query(
            EmotionLog.emotion,
            db.func.count(EmotionLog.id).label('count')
        ).join(EmotionSession).join(Student).join(StudentParent).filter(
            StudentParent.parent_id == user_id,
            db.func.date(EmotionLog.detected_at) >= week_ago
        ).group_by(EmotionLog.emotion).all()
        
        emotion_dict = {item.emotion: item.count for item in emotion_data}
        
        # Hitung trend positif (happy + surprise)
        positive_emotions = (emotion_dict.get('happy', 0) + emotion_dict.get('surprise', 0))
        total_emotions = sum(emotion_dict.values())
        positive_trend = (positive_emotions / total_emotions * 100) if total_emotions > 0 else 0
        
        return jsonify({
            'total_children': total_children,
            'weekly_sessions': weekly_sessions,
            'avg_emotion': 'Happy' if not emotion_dict else max(emotion_dict, key=emotion_dict.get),
            'positive_trend': round(positive_trend, 1),
            'emotion_data': emotion_dict
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/parent/children')
@jwt_required()
@require_role(['orang_tua'])
def get_parent_children():
    """API untuk mendapatkan daftar anak dari orang tua"""
    try:
        from datetime import date, timedelta
        user_id = get_jwt_identity()
        try:
            user_id = int(user_id) if user_id is not None else None
        except Exception:
            pass
        
        # Ambil semua anak dari orang tua ini
        children = db.session.query(Student).join(StudentParent).filter(
            StudentParent.parent_id == user_id,
            Student.is_active == True
        ).all()
        
        # Tambahkan data tambahan untuk setiap anak
        children_data = []
        week_ago = date.today() - timedelta(days=7)
        for child in children:
            child_dict = child.to_dict()
            
            # Hitung sesi minggu ini (handle start_time NULL)
            weekly_sessions = db.session.query(EmotionSession).filter(
                EmotionSession.student_id == child.id,
                EmotionSession.start_time.isnot(None),
                db.func.date(EmotionSession.start_time) >= week_ago
            ).count()
            
            # Ambil emosi terakhir
            last_emotion_log = db.session.query(EmotionLog).filter(
                EmotionLog.student_id == child.id
            ).order_by(EmotionLog.detected_at.desc()).first()
            
            # Hitung skor emosi positif
            positive_count = db.session.query(EmotionLog).join(EmotionSession).filter(
                EmotionSession.student_id == child.id,
                EmotionLog.emotion.in_(['happy', 'surprise'])
            ).count()
            total_count = db.session.query(EmotionLog).join(EmotionSession).filter(
                EmotionSession.student_id == child.id
            ).count()
            avg_emotion_score = (positive_count / total_count * 100) if total_count > 0 else 0
            
            # Sesi terakhir
            last_session = db.session.query(EmotionSession).filter(
                EmotionSession.student_id == child.id
            ).order_by(EmotionSession.start_time.desc()).first()
            last_session_str = None
            try:
                if last_session and last_session.start_time:
                    last_session_str = last_session.start_time.strftime('%d/%m/%Y')
            except Exception:
                last_session_str = None
            
            child_dict.update({
                'weekly_sessions': weekly_sessions,
                'last_emotion': last_emotion_log.emotion if last_emotion_log else None,
                'avg_emotion_score': round(avg_emotion_score, 1),
                'last_session': last_session_str
            })
            
            children_data.append(child_dict)
        
        return jsonify(children_data)
        
    except Exception as e:
        app.logger.exception('get_parent_children failed')
        return jsonify({'error': str(e)}), 500

@app.route('/api/parent/distribution')
@jwt_required()
@require_role(['orang_tua'])
def parent_distribution():
    """Agregasi distribusi emosi untuk seluruh anak milik orang tua (periode hari)."""
    try:
        from datetime import date, timedelta
        user_id = get_jwt_identity()
        user_id = int(user_id) if user_id is not None else None
        period = int(request.args.get('period', 7))
        start_date = date.today() - timedelta(days=period)

        # Ambil semua id anak milik parent
        child_ids = [row[0] for row in db.session.query(Student.id).join(StudentParent).filter(
            StudentParent.parent_id == user_id
        ).all()]

        if not child_ids:
            return jsonify({'distribution': {}, 'per_child': {}, 'period': period}), 200

        # Agregasi total
        rows = db.session.query(
            EmotionLog.emotion,
            db.func.count(EmotionLog.id)
        ).filter(
            EmotionLog.student_id.in_(child_ids),
            db.func.date(EmotionLog.detected_at) >= start_date
        ).group_by(EmotionLog.emotion).all()

        distribution = {r[0]: int(r[1]) for r in rows}

        # Agregasi per anak
        per_child = {}
        for sid in child_ids:
            r = db.session.query(
                EmotionLog.emotion,
                db.func.count(EmotionLog.id)
            ).filter(
                EmotionLog.student_id == sid,
                db.func.date(EmotionLog.detected_at) >= start_date
            ).group_by(EmotionLog.emotion).all()
            per_child[str(sid)] = {x[0]: int(x[1]) for x in r}

        return jsonify({'distribution': distribution, 'per_child': per_child, 'period': period}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/parent/reports/<int:child_id>')
@jwt_required()
@require_role(['orang_tua'])
def get_parent_reports(child_id, period=7):
    """API untuk mendapatkan laporan emosi anak"""
    try:
        user_id = get_jwt_identity()
        user_id = int(user_id) if user_id is not None else None
        period = int(request.args.get('period', 7))
        
        # Cek apakah anak ini adalah anak dari orang tua ini
        child_parent = db.session.query(StudentParent).filter(
            StudentParent.parent_id == user_id,
            StudentParent.student_id == child_id
        ).first()
        
        if not child_parent:
            return jsonify({'error': 'Anda tidak berhak mengakses data anak ini'}), 403
        
        # Ambil data emosi untuk periode tertentu
        from datetime import datetime, date, timedelta
        start_date = date.today() - timedelta(days=period)
        
        emotion_logs = db.session.query(
            EmotionLog.emotion,
            EmotionLog.detected_at,
            EmotionSession.session_name
        ).join(EmotionSession).filter(
            EmotionSession.student_id == child_id,
            db.func.date(EmotionLog.detected_at) >= start_date
        ).order_by(EmotionLog.detected_at.desc()).all()
        
        # Format timeline data
        timeline = []
        for log in emotion_logs:
            timeline.append({
                'date': log.detected_at.strftime('%d/%m/%Y'),
                'time': log.detected_at.strftime('%H:%M'),
                'emotion': log.emotion,
                'session_name': log.session_name
            })
        
        return jsonify({
            'timeline': timeline,
            'period': period,
            'total_records': len(timeline)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/parent/child/<int:child_id>/distribution')
@jwt_required()
@require_role(['orang_tua'])
def parent_child_distribution(child_id):
    """Distribusi emosi dan timeline ringkas per anak untuk orang tua (periode hari)."""
    try:
        from datetime import date, timedelta
        user_id = get_jwt_identity()
        user_id = int(user_id) if user_id is not None else None
        period = int(request.args.get('period', 7))
        start_date = date.today() - timedelta(days=period)

        # Authorization: child must belong to this parent
        allowed = db.session.query(StudentParent).filter(
            StudentParent.parent_id == user_id,
            StudentParent.student_id == child_id
        ).first()
        if not allowed:
            return jsonify({'error': 'Anda tidak berhak mengakses data anak ini'}), 403

        # Distribution
        rows = db.session.query(
            EmotionLog.emotion,
            db.func.count(EmotionLog.id)
        ).filter(
            EmotionLog.student_id == child_id,
            db.func.date(EmotionLog.detected_at) >= start_date
        ).group_by(EmotionLog.emotion).all()
        distribution = {r[0]: int(r[1]) for r in rows}

        # Timeline (ringkas: tanggal dan hitung dominan per hari)
        logs = db.session.query(
            db.func.date(EmotionLog.detected_at).label('d'),
            EmotionLog.emotion,
            db.func.count(EmotionLog.id).label('c')
        ).filter(
            EmotionLog.student_id == child_id,
            db.func.date(EmotionLog.detected_at) >= start_date
        ).group_by('d', EmotionLog.emotion).order_by('d').all()

        per_day = {}
        for d, emo, c in logs:
            d_str = d.isoformat()
            m = per_day.setdefault(d_str, {})
            m[emo] = int(c)

        # Dominan per hari
        timeline = []
        for d_str in sorted(per_day.keys()):
            emo_counts = per_day[d_str]
            dominant = max(emo_counts, key=emo_counts.get) if emo_counts else None
            timeline.append({'date': d_str, 'dominant': dominant, 'counts': emo_counts})

        return jsonify({'distribution': distribution, 'timeline': timeline, 'period': period}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# API Routes untuk Dashboard Admin
@app.route('/api/dashboard/admin/stats')
@jwt_required()
@require_role(['admin'])
def admin_dashboard_stats():
    """API untuk statistik dashboard admin"""
    try:
        # Hitung total users
        total_users = User.query.count()
        
        # Hitung total siswa
        total_students = Student.query.count()
        
        # Hitung total sesi
        total_sessions = EmotionSession.query.count()
        
        # Hitung total deteksi
        total_detections = EmotionLog.query.count()
        
        # Data emosi 7 hari terakhir
        from datetime import datetime, date, timedelta
        week_ago = date.today() - timedelta(days=7)
        emotion_data = db.session.query(
            EmotionLog.emotion,
            db.func.count(EmotionLog.id).label('count')
        ).filter(
            db.func.date(EmotionLog.detected_at) >= week_ago
        ).group_by(EmotionLog.emotion).all()
        
        emotion_dict = {item.emotion: item.count for item in emotion_data}
        
        return jsonify({
            'total_users': total_users,
            'total_students': total_students,
            'total_sessions': total_sessions,
            'total_detections': total_detections,
            'emotion_data': emotion_dict
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/users')
@jwt_required()
@require_role(['admin'])
def get_all_users():
    """API untuk mendapatkan semua users (admin only)"""
    try:
        users = User.query.all()
        return jsonify([user.to_dict() for user in users])
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/students')
@jwt_required()
@require_role(['admin'])
def get_all_students():
    """API untuk mendapatkan semua siswa (admin only)"""
    try:
        students = Student.query.all()
        return jsonify([student.to_dict() for student in students])
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/sessions')
@jwt_required()
@require_role(['admin'])
def get_all_sessions():
    """API untuk mendapatkan semua sesi (admin only)"""
    try:
        sessions = db.session.query(
            EmotionSession.id,
            EmotionSession.session_name,
            EmotionSession.status,
            EmotionSession.start_time,
            EmotionSession.end_time,
            Student.full_name.label('student_name'),
            User.full_name.label('teacher_name'),
            db.func.count(EmotionLog.id).label('total_detections')
        ).outerjoin(Student, EmotionSession.student_id == Student.id
        ).outerjoin(User, EmotionSession.teacher_id == User.id
        ).outerjoin(EmotionLog, EmotionSession.id == EmotionLog.session_id
        ).group_by(EmotionSession.id).all()
        
        sessions_data = []
        for session in sessions:
            sessions_data.append({
                'id': session.id,
                'session_name': session.session_name,
                'status': session.status,
                'start_time': session.start_time.isoformat() if session.start_time else None,
                'end_time': session.end_time.isoformat() if session.end_time else None,
                'student_name': session.student_name,
                'teacher_name': session.teacher_name,
                'total_detections': session.total_detections
            })
        
        return jsonify(sessions_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/students/<int:student_id>/faces/upload', methods=['POST'])
@jwt_required()
@require_role(['guru', 'admin'])
def upload_student_face(student_id):
    """Upload foto wajah siswa untuk dikenali (tersimpan di known_faces/<student_code>/)."""
    try:
        student = Student.query.get(student_id)
        if not student:
            return jsonify({'error': 'Siswa tidak ditemukan'}), 404
        if 'file' not in request.files:
            return jsonify({'error': 'File tidak ditemukan'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Nama file kosong'}), 400
        
        # Validasi file
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
        if '.' not in file.filename or file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
            return jsonify({'error': 'Format file tidak didukung. Gunakan PNG, JPG, JPEG, GIF, atau BMP'}), 400
        
        # Validasi ukuran file (max 5MB)
        file.seek(0, 2)  # Go to end of file
        file_size = file.tell()
        file.seek(0)  # Reset to beginning
        if file_size > 5 * 1024 * 1024:  # 5MB
            return jsonify({'error': 'Ukuran file terlalu besar. Maksimal 5MB'}), 400
        
        # Validasi dimensi gambar
        try:
            from PIL import Image
            import io
            image_data = file.read()
            file.seek(0)  # Reset for saving later
            img = Image.open(io.BytesIO(image_data))
            
            # Validasi dimensi minimum dan maksimum
            width, height = img.size
            if width < 100 or height < 100:
                return jsonify({'error': 'Dimensi gambar terlalu kecil. Minimal 100x100 pixel'}), 400
            if width > 4000 or height > 4000:
                return jsonify({'error': 'Dimensi gambar terlalu besar. Maksimal 4000x4000 pixel'}), 400
                
            # Validasi format gambar
            if img.format not in ['PNG', 'JPEG', 'GIF', 'BMP']:
                return jsonify({'error': 'Format gambar tidak valid'}), 400
                
        except ImportError:
            # PIL tidak tersedia, skip validasi dimensi
            pass
        except Exception as e:
            return jsonify({'error': f'File gambar tidak valid: {str(e)}'}), 400
        
        # Buat folder siswa
        student_dir = os.path.join(KNOWN_FACES_DIR, student.student_code)
        os.makedirs(student_dir, exist_ok=True)
        
        # Generate unique filename
        import uuid
        file_ext = file.filename.rsplit('.', 1)[1].lower()
        unique_filename = f"{uuid.uuid4().hex}.{file_ext}"
        save_path = os.path.join(student_dir, unique_filename)
        
        # Kompresi dan optimasi gambar
        try:
            from PIL import Image
            import io
            
            # Baca gambar
            image_data = file.read()
            file.seek(0)  # Reset for potential retry
            img = Image.open(io.BytesIO(image_data))
            
            # Konversi ke RGB jika perlu (untuk JPEG)
            if img.mode in ('RGBA', 'LA', 'P'):
                img = img.convert('RGB')
            
            # Resize jika terlalu besar (max 800x800 untuk known faces)
            max_size = 800
            if img.width > max_size or img.height > max_size:
                img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
            # Simpan dengan kompresi
            if file_ext in ['jpg', 'jpeg']:
                img.save(save_path, 'JPEG', quality=85, optimize=True)
            else:
                img.save(save_path, optimize=True)
                
        except ImportError:
            # PIL tidak tersedia, simpan file asli
            file.save(save_path)
        except Exception as e:
            # Fallback ke simpan file asli
            file.save(save_path)
            print(f"Warning: Image compression failed, saved original: {e}")
        
        return jsonify({'message': 'Foto berhasil diupload', 'path': save_path, 'filename': unique_filename}), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/students/<int:student_id>/faces', methods=['GET'])
@jwt_required()
@require_role(['guru', 'admin'])
def get_student_faces(student_id):
    """Get daftar foto wajah siswa"""
    try:
        student = Student.query.get(student_id)
        if not student:
            return jsonify({'error': 'Siswa tidak ditemukan'}), 404
        
        student_dir = os.path.join(KNOWN_FACES_DIR, student.student_code)
        if not os.path.exists(student_dir):
            return jsonify({'faces': []}), 200
        
        faces = []
        for filename in os.listdir(student_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                file_path = os.path.join(student_dir, filename)
                file_size = os.path.getsize(file_path)
                file_mtime = os.path.getmtime(file_path)
                
                faces.append({
                    'filename': filename,
                    'size': file_size,
                    'modified': datetime.fromtimestamp(file_mtime).isoformat(),
                    'url': f'/api/students/{student_id}/faces/{filename}'
                })
        
        # Sort by modification time (newest first)
        faces.sort(key=lambda x: x['modified'], reverse=True)
        
        return jsonify({'faces': faces}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/students/<int:student_id>/faces/<filename>', methods=['GET'])
@jwt_required()
@require_role(['guru', 'admin'])
def get_student_face_image(student_id, filename):
    """Get foto wajah siswa"""
    try:
        student = Student.query.get(student_id)
        if not student:
            return jsonify({'error': 'Siswa tidak ditemukan'}), 404
        
        student_dir = os.path.join(KNOWN_FACES_DIR, student.student_code)
        file_path = os.path.join(student_dir, filename)
        
        if not os.path.exists(file_path):
            return jsonify({'error': 'File tidak ditemukan'}), 404
        
        from flask import send_file
        return send_file(file_path)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/students/<int:student_id>/faces/<filename>', methods=['DELETE'])
@jwt_required()
@require_role(['guru', 'admin'])
def delete_student_face(student_id, filename):
    """Hapus foto wajah siswa"""
    try:
        student = Student.query.get(student_id)
        if not student:
            return jsonify({'error': 'Siswa tidak ditemukan'}), 404
        
        student_dir = os.path.join(KNOWN_FACES_DIR, student.student_code)
        file_path = os.path.join(student_dir, filename)
        
        if not os.path.exists(file_path):
            return jsonify({'error': 'File tidak ditemukan'}), 404
        
        # Hapus file
        os.remove(file_path)
        
        # Jika folder kosong, hapus juga
        try:
            if not os.listdir(student_dir):
                os.rmdir(student_dir)
        except Exception:
            pass
        
        return jsonify({'message': 'Foto berhasil dihapus'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/parents/<int:parent_id>/link-student', methods=['POST'])
@jwt_required()
@require_role(['admin', 'guru'])
def link_parent_student(parent_id):
    """Buat relasi orang tua ke siswa."""
    try:
        data = request.get_json() or {}
        student_code = data.get('student_code')
        relationship = data.get('relationship', 'wali')
        is_primary = bool(data.get('is_primary', False))
        if not student_code:
            return jsonify({'error': 'student_code harus diisi'}), 400
        parent = User.query.get(parent_id)
        if not parent or parent.role != 'orang_tua':
            return jsonify({'error': 'Parent tidak ditemukan atau bukan role orang_tua'}), 404
        student = Student.query.filter_by(student_code=student_code).first()
        if not student:
            return jsonify({'error': 'Siswa tidak ditemukan'}), 404
        # Upsert-like: cek existing
        existing = StudentParent.query.filter_by(student_id=student.id, parent_id=parent.id).first()
        if existing:
            existing.relationship = relationship
            existing.is_primary = is_primary
        else:
            sp = StudentParent(student_id=student.id, parent_id=parent.id, relationship=relationship, is_primary=is_primary)
            db.session.add(sp)
        db.session.commit()
        return jsonify({'message': 'Relasi orang tua-siswa berhasil disimpan'}), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/flush-redis', methods=['POST'])
@jwt_required()
@require_role(['admin'])
def flush_redis_manual():
    """Manual flush Redis aggregation ke database (admin only)"""
    try:
        if not redis_client:
            return jsonify({'error': 'Redis tidak tersedia'}), 503
        
        from redis_flush_job import flush_redis_to_db
        flush_redis_to_db()
        
        return jsonify({'message': 'Redis aggregation berhasil di-flush ke database'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analytics/aggregation/<int:teacher_id>')
@jwt_required()
@require_role(['guru', 'admin'])
def get_teacher_aggregation(teacher_id):
    """Get aggregation data untuk teacher (dari Redis + DB)"""
    try:
        user_id = get_jwt_identity()
        user_id = int(user_id) if user_id is not None else None
        role = User.query.get(user_id).role if user_id else None
        
        # Cek permission
        if role != 'admin' and user_id != teacher_id:
            return jsonify({'error': 'Tidak berhak mengakses data ini'}), 403
        
        # Ambil data dari database
        from models import EmotionAggregation
        db_aggregations = EmotionAggregation.query.filter_by(teacher_id=teacher_id).all()
        
        # Ambil data dari Redis (jika tersedia)
        redis_data = {}
        if redis_client:
            try:
                today_str = datetime.utcnow().strftime('%Y-%m-%d')
                pattern = f"emagg:{teacher_id}:*"
                keys = redis_client.keys(pattern)
                
                for key in keys:
                    date_str = key.split(':')[2]
                    emotion_counts = redis_client.hgetall(key)
                    if date_str not in redis_data:
                        redis_data[date_str] = {}
                    redis_data[date_str].update(emotion_counts)
            except Exception:
                pass
        
        # Merge data
        result = {}
        for agg in db_aggregations:
            date_str = agg.date.isoformat()
            if date_str not in result:
                result[date_str] = {}
            result[date_str][agg.emotion] = agg.count
        
        # Tambahkan data Redis
        for date_str, emotions in redis_data.items():
            if date_str not in result:
                result[date_str] = {}
            for emotion, count in emotions.items():
                current_count = result[date_str].get(emotion, 0)
                result[date_str][emotion] = current_count + int(count)
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)

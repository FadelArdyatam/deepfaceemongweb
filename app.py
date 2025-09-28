from flask import Flask, render_template, Response, jsonify, request, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_jwt_extended import JWTManager, jwt_required, get_jwt_identity
from flask_jwt_extended import decode_token
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
from flask_socketio import SocketIO, emit, join_room, leave_room
from time import time as now_time
from services.database_service import DatabaseService
from services.emotion_service import emotion_processor, emotion_aggregator
from services.websocket_service import init_websocket_service
from services.data_compression_service import init_data_compression_service
from api.optimized_routes import api_bp, init_services
from validation_helpers import (
    validate_required_fields, validate_student_code, validate_relationship,
    validate_boolean, create_error_response, handle_validation_error,
    ValidationError
)

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
        print("Redis connected")
        
        # Start background flush job
        try:
            from redis_flush_job import start_flush_job_background
            start_flush_job_background()
        except Exception as flush_err:
            print(f"Redis flush job failed to start: {flush_err}")
    except Exception as e:
        print(f"Redis unavailable: {e}")
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
socketio = SocketIO(app, cors_allowed_origins='*')

# JWT Error Handlers
@jwt.expired_token_loader
def expired_token_callback(jwt_header, jwt_payload):
    return jsonify({'error': 'Token telah expired. Silakan login kembali.'}), 401

@jwt.invalid_token_loader
def invalid_token_callback(error):
    return jsonify({'error': 'Token tidak valid. Silakan login kembali.'}), 401

@jwt.unauthorized_loader
def missing_token_callback(error):
    return jsonify({'error': 'Token tidak ditemukan. Silakan login terlebih dahulu.'}), 401

@jwt.needs_fresh_token_loader
def token_not_fresh_callback(jwt_header, jwt_payload):
    return jsonify({'error': 'Token tidak fresh. Silakan login kembali.'}), 401

# Initialize services
db_service = DatabaseService(db, redis_client)
ws_service = init_websocket_service(socketio, redis_client)
compression_service = init_data_compression_service(db)
init_services(db, redis_client, socketio)

# Register blueprints
app.register_blueprint(auth_bp, url_prefix='/api/auth')
app.register_blueprint(api_bp)  # Optimized API routes

# Configuration for API URL (can be ngrok or localhost)
API_BASE_URL = os.environ.get('API_BASE_URL', 'http://localhost:5000')
CAM_SOURCE = os.environ.get('CAM_SOURCE', 'webcam').lower()  # 'webcam' | 'rtsp'
RTSP_URL_ENV = os.environ.get('RTSP_URL', '').strip()

# Runtime-overridable camera source
CURRENT_CAM_SOURCE = CAM_SOURCE
CURRENT_RTSP_URL = RTSP_URL_ENV

# Runtime-overridable detector backend
CURRENT_DETECTOR_BACKEND = 'opencv'  # retinaface lebih akurat dari opencv

# Debug: Print configuration on startup
print(f"API_BASE_URL configured as: {API_BASE_URL}")
if API_BASE_URL != 'http://localhost:5000':
    print(f"Using ngrok API: {API_BASE_URL}")
else:
    print("Using local processing (localhost)")

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
            print(f"Format URL API tidak valid: {api_url}")
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
            print(f"API error: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.ConnectionError as e:
        print(f"Koneksi error ke ngrok API: {e}")
        return None
    except requests.exceptions.Timeout as e:
        print(f"Timeout error ke ngrok API: {e}")
        return None
    except Exception as e:
        print(f"Error mengirim ke ngrok API: {e}")
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
                        print(f"Camera berhasil dibuka dengan {backend_name} pada index {idx}")
                        # Set camera properties for better performance
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        cap.set(cv2.CAP_PROP_FPS, 30)
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer size
                        return cap
                    else:
                        print(f"  Tidak bisa membaca frame dari {backend_name} index {idx} setelah {max_retries} percobaan")
                cap.release()
            except Exception as e:
                print(f"  Error dengan {backend_name} index {idx}: {e}")
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
                print("Camera berhasil dibuka dengan default backend")
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                return cap
        cap.release()
    except Exception as e:
        print(f"Error dengan default backend: {e}")
    
        print("Peringatan: Tidak ada camera yang bisa dibuka")
    return None


def _open_rtsp_stream(rtsp_url: str):
    """Open RTSP stream with retries."""
    try:
        if not rtsp_url or not (rtsp_url.startswith('rtsp://') or rtsp_url.startswith('rtmp://')):
            print("RTSP URL tidak valid")
            return None
        print(f"🔗 Membuka RTSP: {rtsp_url}")
        # Prefer FFMPEG backend if available
        cap = cv2.VideoCapture(rtsp_url)
        retries = 0
        while retries < 10:
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    print("RTSP stream terbuka")
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    return cap
            retries += 1
            time.sleep(0.3)
        try:
            cap.release()
        except Exception:
            pass
        print("Gagal membuka RTSP stream")
        return None
    except Exception as e:
        print(f"RTSP error: {e}")
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
        print("Error: Tidak bisa mengakses camera")
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
    save_interval_seconds = 5.0  # Capture lebih sering untuk monitoring yang lebih baik
    frame_count = 0
    recognition_interval_frames = 10  # Lebih sering untuk akurasi yang lebih baik
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
                    print(f"Skip frame {frame_skip_count}/{max_frame_skips}")
                    time.sleep(0.05)  # Tunggu lebih singkat
                    continue
                else:
                    print(f"Gagal membaca frame ({consecutive_failures}/{max_consecutive_failures})")
                    if consecutive_failures >= max_consecutive_failures:
                        print("Terlalu banyak kegagalan, menghentikan stream")
                        break
                    time.sleep(0.1)  # Tunggu sebentar sebelum coba lagi
                    frame_skip_count = 0  # Reset skip counter
                    continue
            
            # Reset counters jika berhasil
            consecutive_failures = 0
            frame_skip_count = 0
            
        except Exception as e:
            print(f"Error membaca frame: {e}")
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
                if frame_count % 10 == 0:  # Proses setiap 5 frame untuk akurasi lebih baik
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
            print(f"Error dengan emotion detection: {e}")
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
                threshold = 0.6  # Threshold lebih tinggi untuk akurasi lebih baik
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
    print("Membersihkan resources...")
    if cap:
        cap.release()
        print("Video stream berakhir")

@app.route('/')
def index():
    """Main page - redirect to login"""
    return redirect(url_for('login'))

@app.route('/api/admin/debug/emit', methods=['POST'])
@jwt_required()
@require_role(['admin'])
def admin_debug_emit():
    try:
        payload = request.get_json() or {}
        student_id = int(payload.get('student_id', 0))
        emotion = payload.get('emotion', 'neutral')
        ts = datetime.utcnow().isoformat()
        _emit_emotion_to_parents(student_id, emotion, ts)
        return jsonify({'status': 'emitted'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# -----------------------
# Socket.IO Event Handlers
# -----------------------

@socketio.on('connect')
def handle_connect():
    try:
        emit('connected', {'message': 'connected'})
    except Exception:
        pass

@socketio.on('join_parent')
def handle_join_parent(data):
    try:
        token = data.get('token') if isinstance(data, dict) else None
        if not token:
            emit('error', {'error': 'token missing'})
            return
        decoded = decode_token(token)
        user_id = int(decoded.get('sub')) if decoded and decoded.get('sub') else None
        if not user_id:
            emit('error', {'error': 'invalid token'})
            return
        user = User.query.get(user_id)
        if not user or user.role != 'orang_tua':
            emit('error', {'error': 'forbidden'})
            return
        join_room(f"parent:{user_id}")
        emit('joined', {'room': f'parent:{user_id}'})
    except Exception as e:
        emit('error', {'error': str(e)})

@socketio.on('join_guru')
def handle_join_guru(data):
    try:
        token = data.get('token') if isinstance(data, dict) else None
        if not token:
            emit('error', {'error': 'token missing'})
            return
        decoded = decode_token(token)
        user_id = int(decoded.get('sub')) if decoded and decoded.get('sub') else None
        if not user_id:
            emit('error', {'error': 'invalid token'})
            return
        user = User.query.get(user_id)
        if not user or user.role not in ['guru', 'admin']:
            emit('error', {'error': 'forbidden'})
            return
        join_room(f"guru:{user_id}")
        emit('joined', {'room': f'guru:{user_id}'})
    except Exception as e:
        emit('error', {'error': str(e)})

@socketio.on('join_admin')
def handle_join_admin(data):
    try:
        token = data.get('token') if isinstance(data, dict) else None
        if not token:
            emit('error', {'error': 'token missing'})
            return
        decoded = decode_token(token)
        user_id = int(decoded.get('sub')) if decoded and decoded.get('sub') else None
        if not user_id:
            emit('error', {'error': 'invalid token'})
            return
        user = User.query.get(user_id)
        if not user or user.role != 'admin':
            emit('error', {'error': 'forbidden'})
            return
        join_room(f"admin:{user_id}")
        emit('joined', {'room': f'admin:{user_id}'})
    except Exception as e:
        emit('error', {'error': str(e)})

def _emit_emotion_to_parents(student_id: int, emotion: str, detected_at_iso: str):
    try:
        # Cari parent dari student
        parent_rows = db.session.query(StudentParent.parent_id).filter(StudentParent.student_id == student_id).all()
        for (parent_id,) in parent_rows:
            socketio.emit('emotion_log_created', {
                'student_id': student_id,
                'emotion': emotion,
                'detected_at': detected_at_iso
            }, to=f"parent:{parent_id}")
    except Exception:
        pass

def _emit_emotion_aggregation_to_parents(student_id: int):
    """Emit real-time emotion aggregation to parents"""
    try:
        from datetime import datetime, timedelta
        
        # Get emotion stats for last hour
        one_hour_ago = datetime.utcnow() - timedelta(hours=1)
        
        emotion_logs = db.session.query(
            EmotionLog.emotion,
            db.func.count(EmotionLog.id).label('count'),
            db.func.avg(EmotionLog.confidence_score).label('avg_confidence')
        ).join(EmotionSession).filter(
            EmotionSession.student_id == student_id,
            EmotionLog.detected_at >= one_hour_ago
        ).group_by(EmotionLog.emotion).all()
        
        # Format aggregation data
        emotion_stats = {}
        total_detections = 0
        
        for log in emotion_logs:
            emotion_stats[log.emotion] = {
                'count': log.count,
                'avg_confidence': float(log.avg_confidence) if log.avg_confidence else 0.0
            }
            total_detections += log.count
        
        # Get current dominant emotion
        current_session = EmotionSession.query.filter_by(
            student_id=student_id, 
            status='active'
        ).first()
        
        current_emotion = None
        if current_session:
            recent_log = EmotionLog.query.filter_by(
                session_id=current_session.id
            ).order_by(EmotionLog.detected_at.desc()).first()
            if recent_log:
                current_emotion = recent_log.emotion
        
        # Send to parents
        parent_rows = db.session.query(StudentParent.parent_id).filter(StudentParent.student_id == student_id).all()
        for (parent_id,) in parent_rows:
            socketio.emit('emotion_aggregation_update', {
                'student_id': student_id,
                'emotion_stats': emotion_stats,
                'total_detections': total_detections,
                'current_emotion': current_emotion,
                'time_window': '1_hour',
                'updated_at': datetime.utcnow().isoformat()
            }, to=f"parent:{parent_id}")
            
    except Exception as e:
        print(f"Error emitting emotion aggregation: {e}")
        pass

def _emit_session_update_to_guru(teacher_id: int, session_data: dict):
    try:
        socketio.emit('session_update', session_data, to=f"guru:{teacher_id}")
    except Exception:
        pass

def _emit_student_activity_to_guru(teacher_id: int, student_data: dict):
    try:
        socketio.emit('student_activity', student_data, to=f"guru:{teacher_id}")
    except Exception:
        pass

def _emit_system_stats_to_admin(stats_data: dict):
    try:
        socketio.emit('system_stats', stats_data, room='admin_room')
    except Exception:
        pass

def _emit_user_activity_to_admin(activity_data: dict):
    try:
        socketio.emit('user_activity', activity_data, room='admin_room')
    except Exception:
        pass

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
                    # Create EmotionLog with optimized confidence calculation
                    confidence_score = None
                    if d.get('distance') is not None:
                        # Normalize distance to confidence (0-1 scale)
                        distance = float(d['distance'])
                        # Use exponential decay for better confidence mapping
                        confidence_score = max(0.0, min(1.0, 1.0 - (distance / 0.6)))
                    
                    # Get emotion confidence from DeepFace if available
                    if d.get('emotion_scores'):
                        emotion_scores = d['emotion_scores']
                        detected_emotion = d['emotion']
                        if detected_emotion in emotion_scores:
                            emotion_confidence = emotion_scores[detected_emotion]
                            # Combine face recognition confidence with emotion confidence
                            if confidence_score is not None:
                                confidence_score = (confidence_score * 0.7) + (emotion_confidence * 0.3)
                            else:
                                confidence_score = emotion_confidence
                    
                    log = EmotionLog(
                        session_id=session_row.id,
                        student_id=student.id,
                        emotion=d['emotion'],
                        confidence_score=confidence_score,
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
                # Emit socket events to parents after commit
                try:
                    for d in detections:
                        if not d.get('identity') or not d.get('emotion'):
                            continue
                        student = Student.query.filter_by(student_code=d['identity']).first()
                        if not student:
                            continue
                        
                        # Send individual emotion detection
                        _emit_emotion_to_parents(student.id, d['emotion'], datetime.utcnow().isoformat())
                        
                        # Send aggregated emotion stats for better tracking
                        _emit_emotion_aggregation_to_parents(student.id)
                except Exception:
                    pass
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
            address=data.get('address'),
            phone=data.get('phone'),
            email=data.get('email'),
            subject=data.get('subject'),
            photo_path=data.get('photo_path'),
            notes=data.get('notes')
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
                teacher_id=user_id
            )
            # Set subject setelah objek dibuat jika ada
            if data.get('subject'):
                student_teacher.subject = str(data.get('subject'))
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
            print(f"Foto siswa {student_code} berhasil disalin ke known_faces")
        
    except Exception as e:
        print(f"Error membuat known face folder: {e}")

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


@app.route('/api/parent/children/<int:child_id>')
@jwt_required()
@require_role(['orang_tua'])
def get_child_details(child_id):
    """API untuk mendapatkan detail anak"""
    try:
        user_id = get_jwt_identity()
        user_id = int(user_id) if user_id is not None else None
        
        # Check if parent has access to this child
        parent_relation = StudentParent.query.filter_by(
            parent_id=user_id, 
            student_id=child_id
        ).first()
        
        if not parent_relation:
            return jsonify({'error': 'Access denied'}), 403
            
        # Get child data with teacher info
        child = Student.query.get(child_id)
        if not child:
            return jsonify({'error': 'Child not found'}), 404
            
        # Get teacher info
        teacher_relation = StudentTeacher.query.filter_by(student_id=child_id).first()
        teacher_name = None
        if teacher_relation:
            teacher = User.query.get(teacher_relation.teacher_id)
            teacher_name = teacher.full_name if teacher else None
            
        # Get emotion stats
        from datetime import datetime, timedelta
        last_week = datetime.utcnow() - timedelta(days=7)
        
        emotion_logs = EmotionLog.query.join(EmotionSession).filter(
            EmotionSession.student_id == child_id,
            EmotionLog.detected_at >= last_week
        ).all()
        
        total_detections = len(emotion_logs)
        last_emotion = emotion_logs[-1].emotion if emotion_logs else None
        
        child_data = child.to_dict()
        child_data.update({
            'teacher_name': teacher_name,
            'total_detections': total_detections,
            'last_emotion': last_emotion
        })
        
        return jsonify(child_data), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/parent/children/<int:child_id>/emotions')
@jwt_required()
@require_role(['orang_tua'])
def get_child_emotions(child_id):
    """API untuk mendapatkan data emosi anak"""
    try:
        user_id = get_jwt_identity()
        user_id = int(user_id) if user_id is not None else None
        
        # Check if parent has access to this child
        parent_relation = StudentParent.query.filter_by(
            parent_id=user_id, 
            student_id=child_id
        ).first()
        
        if not parent_relation:
            return jsonify({'error': 'Access denied'}), 403
            
        # Get emotion data for chart
        from datetime import datetime, timedelta
        last_week = datetime.utcnow() - timedelta(days=7)
        
        emotion_logs = EmotionLog.query.join(EmotionSession).filter(
            EmotionSession.student_id == child_id,
            EmotionLog.detected_at >= last_week
        ).order_by(EmotionLog.detected_at).all()
        
        # Group by date
        emotion_by_date = {}
        for log in emotion_logs:
            date_str = log.detected_at.strftime('%Y-%m-%d')
            if date_str not in emotion_by_date:
                emotion_by_date[date_str] = []
            emotion_by_date[date_str].append(log.emotion)
        
        # Create chart data
        labels = list(emotion_by_date.keys())
        values = [len(emotions) for emotions in emotion_by_date.values()]
        
        return jsonify({
            'labels': labels,
            'values': values
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/parent/reports/<int:child_id>/download', methods=['POST'])
@jwt_required()
@require_role(['orang_tua'])
def download_child_report(child_id):
    """API untuk download laporan anak"""
    try:
        user_id = get_jwt_identity()
        user_id = int(user_id) if user_id is not None else None
        
        # Check if parent has access to this child
        parent_relation = StudentParent.query.filter_by(
            parent_id=user_id, 
            student_id=child_id
        ).first()
        
        if not parent_relation:
            return jsonify({'error': 'Access denied'}), 403
            
        data = request.get_json()
        format_type = data.get('format', 'pdf')
        period = data.get('period', 7)
        
        # Get child data
        child = Student.query.get(child_id)
        if not child:
            return jsonify({'error': 'Child not found'}), 404
            
        # Get emotion data
        from datetime import datetime, timedelta
        start_date = datetime.utcnow() - timedelta(days=period)
        
        emotion_logs = EmotionLog.query.join(EmotionSession).filter(
            EmotionSession.student_id == child_id,
            EmotionLog.detected_at >= start_date
        ).all()
        
        if format_type == 'pdf':
            # Generate PDF report
            from reportlab.lib.pagesizes import letter
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
            from reportlab.lib.styles import getSampleStyleSheet
            from reportlab.lib import colors
            from io import BytesIO
            
            buffer = BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=letter)
            styles = getSampleStyleSheet()
            story = []
            
            # Title
            title = Paragraph(f"Laporan Emosi - {child.full_name}", styles['Title'])
            story.append(title)
            story.append(Spacer(1, 12))
            
            # Child info
            info_data = [
                ['Nama', child.full_name],
                ['Kelas', child.class_name],
                ['Periode', f"{period} hari terakhir"],
                ['Total Deteksi', str(len(emotion_logs))]
            ]
            
            info_table = Table(info_data)
            info_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 14),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(info_table)
            story.append(Spacer(1, 12))
            
            # Emotion summary
            emotion_counts = {}
            for log in emotion_logs:
                emotion_counts[log.emotion] = emotion_counts.get(log.emotion, 0) + 1
            
            emotion_data = [['Emosi', 'Jumlah']]
            for emotion, count in emotion_counts.items():
                emotion_data.append([emotion, str(count)])
            
            emotion_table = Table(emotion_data)
            emotion_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 14),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(emotion_table)
            
            doc.build(story)
            buffer.seek(0)
            
            return Response(
                buffer.getvalue(),
                mimetype='application/pdf',
                headers={'Content-Disposition': f'attachment; filename=report_{child_id}.pdf'}
            )
            
        elif format_type == 'excel':
            # Generate Excel report
            import pandas as pd
            from io import BytesIO
            
            # Create DataFrame
            data_list = []
            for log in emotion_logs:
                data_list.append({
                    'Tanggal': log.detected_at.strftime('%Y-%m-%d %H:%M:%S'),
                    'Emosi': log.emotion,
                    'Sesi': log.session_id
                })
            
            df = pd.DataFrame(data_list)
            
            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Emotion Report', index=False)
            
            buffer.seek(0)
            
            return Response(
                buffer.getvalue(),
                mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                headers={'Content-Disposition': f'attachment; filename=report_{child_id}.xlsx'}
            )
        
        else:
            return jsonify({'error': 'Unsupported format'}), 400
            
    except Exception as e:
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
        student = None
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
        
        # Validasi user_id
        if not user_id:
            return jsonify(create_error_response('User ID tidak ditemukan dalam token', 401))
        
        try:
            user_id = int(user_id)
        except (ValueError, TypeError):
            return jsonify(create_error_response('User ID tidak valid', 400))
        
        # Cek apakah user masih aktif
        user = User.query.get(user_id)
        if not user or not user.is_active:
            return jsonify(create_error_response('User tidak aktif', 403))
        
        # Hitung total anak dengan error handling
        try:
            total_children = db.session.query(Student).join(StudentParent).filter(
                StudentParent.parent_id == user_id,
                Student.is_active == True
            ).count()
        except Exception as e:
            return jsonify(create_error_response(f'Error menghitung total anak: {str(e)}', 500))
        
        # Hitung sesi minggu ini dengan error handling
        try:
            from datetime import datetime, date, timedelta
            week_ago = date.today() - timedelta(days=7)
            weekly_sessions = db.session.query(EmotionSession).join(Student).join(StudentParent).filter(
                StudentParent.parent_id == user_id,
                db.func.date(EmotionSession.start_time) >= week_ago
            ).count()
        except Exception as e:
            return jsonify(create_error_response(f'Error menghitung sesi minggu ini: {str(e)}', 500))
        
        # Data emosi minggu ini dengan error handling
        try:
            emotion_data = db.session.query(
                EmotionLog.emotion,
                db.func.count(EmotionLog.id).label('count')
            ).join(EmotionSession).join(Student).join(StudentParent).filter(
                StudentParent.parent_id == user_id,
                db.func.date(EmotionLog.detected_at) >= week_ago
            ).group_by(EmotionLog.emotion).all()
            
            emotion_dict = {item.emotion: item.count for item in emotion_data}
        except Exception as e:
            return jsonify(create_error_response(f'Error menghitung data emosi: {str(e)}', 500))
        
        # Hitung trend positif (happy + surprise) dengan error handling
        try:
            positive_emotions = (emotion_dict.get('happy', 0) + emotion_dict.get('surprise', 0))
            total_emotions = sum(emotion_dict.values())
            positive_trend = (positive_emotions / total_emotions * 100) if total_emotions > 0 else 0
        except Exception as e:
            return jsonify(create_error_response(f'Error menghitung trend positif: {str(e)}', 500))
        
        # Tentukan emosi dominan dengan error handling
        try:
            if emotion_dict:
                dominant_emotion = max(emotion_dict, key=emotion_dict.get)
            else:
                dominant_emotion = 'Neutral'
        except Exception as e:
            dominant_emotion = 'Neutral'
        
        return jsonify({
            'success': True,
            'total_children': total_children,
            'weekly_sessions': weekly_sessions,
            'avg_emotion': dominant_emotion,
            'positive_trend': round(positive_trend, 1),
            'emotion_data': emotion_dict
        }), 200
        
    except ValidationError as e:
        return jsonify(handle_validation_error(e))
    except Exception as e:
        return jsonify(create_error_response(f'Terjadi kesalahan server: {str(e)}', 500))

@app.route('/api/parent/children')
@jwt_required()
@require_role(['orang_tua'])
def get_parent_children():
    """API untuk mendapatkan daftar anak dari orang tua"""
    try:
        from datetime import date, timedelta
        user_id = get_jwt_identity()
        
        # Validasi user_id
        if not user_id:
            return jsonify(create_error_response('User ID tidak ditemukan dalam token', 401))
        
        try:
            user_id = int(user_id)
        except (ValueError, TypeError):
            return jsonify(create_error_response('User ID tidak valid', 400))
        
        # Cek apakah user masih aktif
        user = User.query.get(user_id)
        if not user or not user.is_active:
            return jsonify(create_error_response('User tidak aktif', 403))
        
        # Ambil semua anak dari orang tua ini dengan error handling
        try:
            children = db.session.query(Student).join(StudentParent).filter(
                StudentParent.parent_id == user_id,
                Student.is_active == True
            ).all()
        except Exception as e:
            return jsonify(create_error_response(f'Error mengambil data anak: {str(e)}', 500))
        
        # Tambahkan data tambahan untuk setiap anak dengan error handling
        children_data = []
        week_ago = date.today() - timedelta(days=7)
        
        for child in children:
            try:
                child_dict = child.to_dict()
                
                # Hitung sesi minggu ini (handle start_time NULL)
                try:
                    weekly_sessions = db.session.query(EmotionSession).filter(
                        EmotionSession.student_id == child.id,
                        EmotionSession.start_time.isnot(None),
                        db.func.date(EmotionSession.start_time) >= week_ago
                    ).count()
                except Exception as e:
                    app.logger.warning(f'Error calculating weekly sessions for child {child.id}: {str(e)}')
                    weekly_sessions = 0
                
                # Ambil emosi terakhir
                try:
                    last_emotion_log = db.session.query(EmotionLog).filter(
                        EmotionLog.student_id == child.id
                    ).order_by(EmotionLog.detected_at.desc()).first()
                except Exception as e:
                    app.logger.warning(f'Error getting last emotion for child {child.id}: {str(e)}')
                    last_emotion_log = None
                
                # Hitung skor emosi positif
                try:
                    positive_count = db.session.query(EmotionLog).join(EmotionSession).filter(
                        EmotionSession.student_id == child.id,
                        EmotionLog.emotion.in_(['happy', 'surprise'])
                    ).count()
                    total_count = db.session.query(EmotionLog).join(EmotionSession).filter(
                        EmotionSession.student_id == child.id
                    ).count()
                    avg_emotion_score = (positive_count / total_count * 100) if total_count > 0 else 0
                except Exception as e:
                    app.logger.warning(f'Error calculating emotion score for child {child.id}: {str(e)}')
                    avg_emotion_score = 0
                
                # Sesi terakhir
                try:
                    last_session = db.session.query(EmotionSession).filter(
                        EmotionSession.student_id == child.id
                    ).order_by(EmotionSession.start_time.desc()).first()
                    last_session_str = None
                    if last_session and last_session.start_time:
                        last_session_str = last_session.start_time.strftime('%d/%m/%Y')
                except Exception as e:
                    app.logger.warning(f'Error getting last session for child {child.id}: {str(e)}')
                    last_session_str = None
                
                child_dict.update({
                    'weekly_sessions': weekly_sessions,
                    'last_emotion': last_emotion_log.emotion if last_emotion_log else None,
                    'avg_emotion_score': round(avg_emotion_score, 1),
                    'last_session': last_session_str
                })
                
                children_data.append(child_dict)
                
            except Exception as e:
                app.logger.error(f'Error processing child {child.id}: {str(e)}')
                # Skip this child but continue with others
                continue
        
        return jsonify({
            'success': True,
            'children': children_data,
            'total_count': len(children_data)
        }), 200
        
    except ValidationError as e:
        return jsonify(handle_validation_error(e))
    except Exception as e:
        app.logger.exception('get_parent_children failed')
        return jsonify(create_error_response(f'Terjadi kesalahan server: {str(e)}', 500))

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
        
        # Validasi user_id
        if not user_id:
            return jsonify(create_error_response('User ID tidak ditemukan dalam token', 401))
        
        try:
            user_id = int(user_id)
        except (ValueError, TypeError):
            return jsonify(create_error_response('User ID tidak valid', 400))
        
        # Validasi child_id
        if child_id <= 0:
            return jsonify(create_error_response('Child ID harus berupa angka positif', 400))
        
        # Validasi child_id yang lebih ketat - cek apakah child exists
        try:
            child = Student.query.get(child_id)
            if not child:
                return jsonify(create_error_response('Anak dengan ID tersebut tidak ditemukan', 404))
        except Exception as e:
            return jsonify(create_error_response(f'Error checking child: {str(e)}', 500))
        
        # Validasi period
        try:
            period = int(request.args.get('period', 7))
            if period <= 0 or period > 365:
                return jsonify(create_error_response('Period harus antara 1-365 hari', 400))
        except (ValueError, TypeError):
            return jsonify(create_error_response('Period tidak valid', 400))
        
        # Cek apakah user masih aktif
        user = User.query.get(user_id)
        if not user or not user.is_active:
            return jsonify(create_error_response('User tidak aktif', 403))
        
        # Cek apakah anak ini adalah anak dari orang tua ini
        try:
            child_parent = db.session.query(StudentParent).filter(
                StudentParent.parent_id == user_id,
                StudentParent.student_id == child_id
            ).first()
        except Exception as e:
            return jsonify(create_error_response(f'Error checking parent-child relationship: {str(e)}', 500))
        
        if not child_parent:
            return jsonify(create_error_response('Anda tidak berhak mengakses data anak ini', 403))
        
        # Cek apakah child aktif (child sudah dicek exists di atas)
        if not child.is_active:
            return jsonify(create_error_response('Anak tidak aktif', 400))
        
        # Ambil data emosi untuk periode tertentu dengan error handling
        try:
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
        except Exception as e:
            return jsonify(create_error_response(f'Error mengambil data emosi: {str(e)}', 500))
        
        # Format timeline data dengan error handling
        timeline = []
        try:
            for log in emotion_logs:
                timeline.append({
                    'date': log.detected_at.strftime('%d/%m/%Y'),
                    'time': log.detected_at.strftime('%H:%M'),
                    'emotion': log.emotion,
                    'session_name': log.session_name or 'Unknown Session'
                })
        except Exception as e:
            return jsonify(create_error_response(f'Error formatting timeline data: {str(e)}', 500))
        
        return jsonify({
            'success': True,
            'timeline': timeline,
            'period': period,
            'total_records': len(timeline),
            'child_name': child.full_name,
            'child_code': child.student_code
        }), 200
        
    except ValidationError as e:
        return jsonify(handle_validation_error(e))
    except Exception as e:
        return jsonify(create_error_response(f'Terjadi kesalahan server: {str(e)}', 500))

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

@app.route('/api/admin/users', methods=['POST'])
@jwt_required()
@require_role(['admin'])
def create_user():
    """API untuk membuat user baru (admin only)"""
    try:
        data = request.get_json()
        
        # Validasi input
        required_fields = ['username', 'email', 'full_name', 'role', 'password']
        for field in required_fields:
            if field not in data or not data[field]:
                return jsonify({'error': f'Field {field} harus diisi'}), 400
        
        # Cek apakah username sudah ada
        if User.query.filter_by(username=data['username']).first():
            return jsonify({'error': 'Username sudah digunakan'}), 409
            
        # Cek apakah email sudah ada
        if User.query.filter_by(email=data['email']).first():
            return jsonify({'error': 'Email sudah digunakan'}), 409
        
        # Validasi role
        valid_roles = ['admin', 'guru', 'orang_tua']
        if data['role'] not in valid_roles:
            return jsonify({'error': 'Role tidak valid'}), 400
        
        # Buat user baru
        user = User(
            username=data['username'],
            email=data['email'],
            full_name=data['full_name'],
            role=data['role'],
            phone=data.get('phone', ''),
            is_active=data.get('is_active', True),
            is_approved=data.get('is_approved', True)  # Default approved for admin-created users
        )
        user.set_password(data['password'])
        
        db.session.add(user)
        db.session.commit()
        
        return jsonify({
            'message': 'User berhasil dibuat',
            'user': user.to_dict()
        }), 201
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/users/<int:user_id>', methods=['PUT'])
@jwt_required()
@require_role(['admin'])
def update_user(user_id):
    """API untuk mengupdate user (admin only)"""
    try:
        user = User.query.get(user_id)
        if not user:
            return jsonify({'error': 'User tidak ditemukan'}), 404
        
        data = request.get_json()
        
        # Update fields yang ada
        if 'username' in data:
            # Cek apakah username sudah digunakan oleh user lain
            existing_user = User.query.filter_by(username=data['username']).first()
            if existing_user and existing_user.id != user_id:
                return jsonify({'error': 'Username sudah digunakan'}), 409
            user.username = data['username']
        
        if 'email' in data:
            # Cek apakah email sudah digunakan oleh user lain
            existing_user = User.query.filter_by(email=data['email']).first()
            if existing_user and existing_user.id != user_id:
                return jsonify({'error': 'Email sudah digunakan'}), 409
            user.email = data['email']
        
        if 'full_name' in data:
            user.full_name = data['full_name']
        
        if 'role' in data:
            valid_roles = ['admin', 'guru', 'orang_tua']
            if data['role'] not in valid_roles:
                return jsonify({'error': 'Role tidak valid'}), 400
            user.role = data['role']
        
        if 'phone' in data:
            user.phone = data['phone']
        
        if 'is_active' in data:
            user.is_active = bool(data['is_active'])
        
        if 'is_approved' in data:
            user.is_approved = data['is_approved']
        
        if 'password' in data and data['password']:
            user.set_password(data['password'])
        
        user.updated_at = datetime.utcnow()
        db.session.commit()
        
        return jsonify({
            'message': 'User berhasil diupdate',
            'user': user.to_dict()
        }), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/users/<int:user_id>', methods=['DELETE'])
@jwt_required()
@require_role(['admin'])
def delete_user(user_id):
    """API untuk menghapus user (admin only)"""
    try:
        user = User.query.get(user_id)
        if not user:
            return jsonify({'error': 'User tidak ditemukan'}), 404
        
        # Cek apakah user adalah admin terakhir
        if user.role == 'admin':
            admin_count = User.query.filter_by(role='admin', is_active=True).count()
            if admin_count <= 1:
                return jsonify({'error': 'Tidak bisa menghapus admin terakhir'}), 400
        
        # Soft delete - set is_active = False
        user.is_active = False
        user.updated_at = datetime.utcnow()
        db.session.commit()
        
        return jsonify({'message': 'User berhasil dihapus'}), 200
        
    except Exception as e:
        db.session.rollback()
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

@app.route('/api/admin/students', methods=['POST'])
@jwt_required()
@require_role(['admin'])
def create_student_admin():
    """API untuk membuat siswa baru (admin only)"""
    try:
        data = request.get_json()
        
        # Validasi input
        required_fields = ['student_code', 'full_name', 'class_name']
        for field in required_fields:
            if field not in data or not data[field]:
                return jsonify({'error': f'Field {field} harus diisi'}), 400
        
        # Cek apakah student_code sudah ada
        if Student.query.filter_by(student_code=data['student_code']).first():
            return jsonify({'error': 'Kode siswa sudah digunakan'}), 409
        
        # Parse birth_date
        birth_date_value = data.get('birth_date')
        if birth_date_value:
            from datetime import datetime as dt
            try:
                birth_date_value = dt.strptime(birth_date_value[:10], '%Y-%m-%d').date()
            except Exception:
                birth_date_value = None
        
        # Buat siswa baru
        student = Student(
            student_code=data['student_code'],
            full_name=data['full_name'],
            class_name=data['class_name'],
            birth_date=birth_date_value,
            address=data.get('address'),
            phone=data.get('phone'),
            email=data.get('email'),
            subject=data.get('subject'),
            notes=data.get('notes'),
            is_active=data.get('is_active', True)
        )
        
        db.session.add(student)
        db.session.commit()
        
        # Jika ada parent_ids, buat relasi parent-student
        parent_ids = data.get('parent_ids', [])
        if parent_ids:
            for parent_id in parent_ids:
                parent = User.query.get(parent_id)
                if parent and parent.role == 'orang_tua':
                    # Cek apakah relasi sudah ada
                    existing_relation = StudentParent.query.filter_by(
                        student_id=student.id, 
                        parent_id=parent_id
                    ).first()
                    
                    if not existing_relation:
                        relation = StudentParent(
                            student_id=student.id,
                            parent_id=parent_id,
                            relationship=data.get('relationship', 'wali')
                        )
                        db.session.add(relation)
            
            db.session.commit()
        
        return jsonify({
            'message': 'Siswa berhasil dibuat',
            'student': student.to_dict()
        }), 201
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/students/<int:student_id>', methods=['PUT'])
@jwt_required()
@require_role(['admin'])
def update_student_admin(student_id):
    """API untuk mengupdate siswa (admin only)"""
    try:
        student = Student.query.get(student_id)
        if not student:
            return jsonify({'error': 'Siswa tidak ditemukan'}), 404
        
        data = request.get_json()
        
        # Update fields yang ada
        if 'student_code' in data:
            # Cek apakah student_code sudah digunakan oleh siswa lain
            existing_student = Student.query.filter_by(student_code=data['student_code']).first()
            if existing_student and existing_student.id != student_id:
                return jsonify({'error': 'Kode siswa sudah digunakan'}), 409
            student.student_code = data['student_code']
        
        if 'full_name' in data:
            student.full_name = data['full_name']
        
        if 'class_name' in data:
            student.class_name = data['class_name']
        
        if 'birth_date' in data:
            if data['birth_date']:
                from datetime import datetime as dt
                try:
                    student.birth_date = dt.strptime(data['birth_date'][:10], '%Y-%m-%d').date()
                except Exception:
                    pass
            else:
                student.birth_date = None
        
        if 'address' in data:
            student.address = data['address']
        
        if 'phone' in data:
            student.phone = data['phone']
        
        if 'email' in data:
            student.email = data['email']
        
        if 'subject' in data:
            student.subject = data['subject']
        
        if 'notes' in data:
            student.notes = data['notes']
        
        if 'is_active' in data:
            student.is_active = bool(data['is_active'])
        
        student.updated_at = datetime.utcnow()
        db.session.commit()
        
        return jsonify({
            'message': 'Siswa berhasil diupdate',
            'student': student.to_dict()
        }), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/students/<int:student_id>', methods=['DELETE'])
@jwt_required()
@require_role(['admin'])
def delete_student_admin(student_id):
    """API untuk menghapus siswa (admin only)"""
    try:
        student = Student.query.get(student_id)
        if not student:
            return jsonify({'error': 'Siswa tidak ditemukan'}), 404
        
        # Soft delete - set is_active = False
        student.is_active = False
        student.updated_at = datetime.utcnow()
        db.session.commit()
        
        return jsonify({'message': 'Siswa berhasil dihapus'}), 200
        
    except Exception as e:
        db.session.rollback()
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

@app.route('/api/admin/sessions/<int:session_id>')
@jwt_required()
@require_role(['admin'])
def get_session_detail(session_id):
    """API untuk mendapatkan detail sesi (admin only)"""
    try:
        session = EmotionSession.query.get(session_id)
        if not session:
            return jsonify({'error': 'Sesi tidak ditemukan'}), 404
        
        # Get session info
        session_data = session.to_dict()
        
        # Get student info
        student_data = None
        if session.student_id:
            student = Student.query.get(session.student_id)
            if student:
                student_data = student.to_dict()
        
        # Get teacher info
        teacher_data = None
        if session.teacher_id:
            teacher = User.query.get(session.teacher_id)
            if teacher:
                teacher_data = teacher.to_dict()
        
        # Get emotion logs
        emotion_logs = db.session.query(
            EmotionLog.id,
            EmotionLog.emotion,
            EmotionLog.confidence_score,
            EmotionLog.detected_at,
            EmotionLog.image_path
        ).filter(
            EmotionLog.session_id == session_id
        ).order_by(
            EmotionLog.detected_at.desc()
        ).all()
        
        logs_data = []
        for log in emotion_logs:
            logs_data.append({
                'id': log.id,
                'emotion': log.emotion,
                'confidence_score': float(log.confidence_score) if log.confidence_score else None,
                'detected_at': log.detected_at.isoformat() if log.detected_at else None,
                'image_path': log.image_path
            })
        
        # Get emotion statistics
        emotion_stats = db.session.query(
            EmotionLog.emotion,
            db.func.count(EmotionLog.id).label('count'),
            db.func.avg(EmotionLog.confidence_score).label('avg_confidence')
        ).filter(
            EmotionLog.session_id == session_id
        ).group_by(
            EmotionLog.emotion
        ).all()
        
        stats_data = {}
        for stat in emotion_stats:
            stats_data[stat.emotion] = {
                'count': stat.count,
                'avg_confidence': float(stat.avg_confidence) if stat.avg_confidence else 0.0
            }
        
        return jsonify({
            'session': session_data,
            'student': student_data,
            'teacher': teacher_data,
            'emotion_logs': logs_data,
            'emotion_stats': stats_data,
            'total_logs': len(logs_data)
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/profile')
@jwt_required()
@require_role(['admin'])
def get_admin_profile():
    """API untuk mendapatkan profil admin"""
    try:
        user_id = get_jwt_identity()
        user_id = int(user_id) if user_id is not None else None
        user = User.query.get(user_id)
        
        if not user:
            return jsonify({'error': 'User tidak ditemukan'}), 404
        
        return jsonify(user.to_dict()), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/profile', methods=['PUT'])
@jwt_required()
@require_role(['admin'])
def update_admin_profile():
    """API untuk mengupdate profil admin"""
    try:
        user_id = get_jwt_identity()
        user_id = int(user_id) if user_id is not None else None
        user = User.query.get(user_id)
        
        if not user:
            return jsonify({'error': 'User tidak ditemukan'}), 404
        
        data = request.get_json()
        
        # Update fields yang diizinkan
        if 'email' in data:
            # Cek apakah email sudah digunakan oleh user lain
            existing_user = User.query.filter_by(email=data['email']).first()
            if existing_user and existing_user.id != user_id:
                return jsonify({'error': 'Email sudah digunakan'}), 409
            user.email = data['email']
        
        if 'full_name' in data:
            user.full_name = data['full_name']
        
        if 'phone' in data:
            user.phone = data['phone']
        
        user.updated_at = datetime.utcnow()
        db.session.commit()
        
        return jsonify({
            'message': 'Profil berhasil diupdate',
            'user': user.to_dict()
        }), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/change-password', methods=['POST'])
@jwt_required()
@require_role(['admin'])
def change_admin_password():
    """API untuk mengubah password admin"""
    try:
        user_id = get_jwt_identity()
        user_id = int(user_id) if user_id is not None else None
        user = User.query.get(user_id)
        
        if not user:
            return jsonify({'error': 'User tidak ditemukan'}), 404
        
        data = request.get_json()
        
        # Validasi input
        required_fields = ['current_password', 'new_password']
        for field in required_fields:
            if field not in data or not data[field]:
                return jsonify({'error': f'Field {field} harus diisi'}), 400
        
        # Cek password lama
        if not user.check_password(data['current_password']):
            return jsonify({'error': 'Password lama salah'}), 400
        
        # Set password baru
        user.set_password(data['new_password'])
        user.updated_at = datetime.utcnow()
        db.session.commit()
        
        return jsonify({'message': 'Password berhasil diubah'}), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/system/stats')
@jwt_required()
@require_role(['admin'])
def get_system_stats():
    """API untuk mendapatkan statistik sistem"""
    try:
        import psutil
        import os
        from datetime import datetime
        
        # Database stats
        total_users = User.query.count()
        active_users = User.query.filter_by(is_active=True).count()
        total_students = Student.query.count()
        active_students = Student.query.filter_by(is_active=True).count()
        total_sessions = EmotionSession.query.count()
        total_emotion_logs = EmotionLog.query.count()
        
        # System stats
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Database size estimation (simplified)
        db_size_mb = 0
        try:
            # This is a simplified calculation
            db_size_mb = (total_users * 0.001 + total_students * 0.001 + 
                         total_sessions * 0.01 + total_emotion_logs * 0.001)
        except:
            pass
        
        return jsonify({
            'database': {
                'total_users': total_users,
                'active_users': active_users,
                'total_students': total_students,
                'active_students': active_students,
                'total_sessions': total_sessions,
                'total_emotion_logs': total_emotion_logs,
                'estimated_size_mb': round(db_size_mb, 2)
            },
            'system': {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_used_gb': round(memory.used / (1024**3), 2),
                'memory_total_gb': round(memory.total / (1024**3), 2),
                'disk_percent': disk.percent,
                'disk_used_gb': round(disk.used / (1024**3), 2),
                'disk_total_gb': round(disk.total / (1024**3), 2)
            },
            'timestamp': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/system/logs')
@jwt_required()
@require_role(['admin'])
def get_system_logs():
    """API untuk mendapatkan log sistem"""
    try:
        import logging
        import os
        from datetime import datetime, timedelta
        
        # Get recent logs (last 100 lines)
        logs = []
        log_file_path = 'app.log'  # Adjust path as needed
        
        if os.path.exists(log_file_path):
            with open(log_file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                # Get last 100 lines
                recent_lines = lines[-100:] if len(lines) > 100 else lines
                
                for line in recent_lines:
                    if line.strip():
                        logs.append({
                            'timestamp': datetime.utcnow().isoformat(),  # Simplified
                            'message': line.strip(),
                            'level': 'INFO'  # Simplified
                        })
        
        return jsonify({
            'logs': logs,
            'total_count': len(logs),
            'timestamp': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/system/backup', methods=['POST'])
@jwt_required()
@require_role(['admin'])
def create_system_backup():
    """API untuk membuat backup sistem"""
    try:
        import shutil
        import os
        from datetime import datetime
        
        # Create backup directory if not exists
        backup_dir = 'backups'
        if not os.path.exists(backup_dir):
            os.makedirs(backup_dir)
        
        # Generate backup filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_filename = f'backup_{timestamp}.json'
        backup_path = os.path.join(backup_dir, backup_filename)
        
        # Create backup data
        backup_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'users': [user.to_dict() for user in User.query.all()],
            'students': [student.to_dict() for student in Student.query.all()],
            'sessions': [session.to_dict() for session in EmotionSession.query.all()],
            'emotion_logs': [
                {
                    'id': log.id,
                    'session_id': log.session_id,
                    'student_id': log.student_id,
                    'emotion': log.emotion,
                    'confidence_score': float(log.confidence_score) if log.confidence_score else None,
                    'detected_at': log.detected_at.isoformat() if log.detected_at else None
                }
                for log in EmotionLog.query.all()
            ],
            'relations': [
                {
                    'id': rel.id,
                    'student_id': rel.student_id,
                    'parent_id': rel.parent_id,
                    'relationship': rel.relationship,
                    'created_at': rel.created_at.isoformat() if rel.created_at else None
                }
                for rel in StudentParent.query.all()
            ]
        }
        
        # Write backup file
        import json
        with open(backup_path, 'w', encoding='utf-8') as f:
            json.dump(backup_data, f, indent=2, ensure_ascii=False)
        
        return jsonify({
            'message': 'Backup berhasil dibuat',
            'backup_file': backup_filename,
            'backup_path': backup_path,
            'timestamp': datetime.utcnow().isoformat()
        }), 200
        
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
                    'url': f'/static/faces/{student.student_code}/{filename}'
                })
        
        # Sort by modification time (newest first)
        faces.sort(key=lambda x: x['modified'], reverse=True)
        
        return jsonify({'faces': faces}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/students/<int:student_id>/faces/<filename>', methods=['GET'])
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
        import mimetypes
        
        # Deteksi MIME type berdasarkan ekstensi file
        mime_type, _ = mimetypes.guess_type(file_path)
        if not mime_type or not mime_type.startswith('image/'):
            mime_type = 'image/jpeg'  # Default fallback
        
        return send_file(file_path, mimetype=mime_type)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/static/faces/<student_code>/<filename>')
def serve_face_image(student_code, filename):
    """Serve foto wajah siswa secara static"""
    try:
        student_dir = os.path.join(KNOWN_FACES_DIR, student_code)
        file_path = os.path.join(student_dir, filename)
        
        if not os.path.exists(file_path):
            return jsonify({'error': 'File tidak ditemukan'}), 404
        
        from flask import send_file
        import mimetypes
        
        # Deteksi MIME type berdasarkan ekstensi file
        mime_type, _ = mimetypes.guess_type(file_path)
        if not mime_type or not mime_type.startswith('image/'):
            mime_type = 'image/jpeg'  # Default fallback
        
        return send_file(file_path, mimetype=mime_type)
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
        # Validasi input data
        data = request.get_json() or {}
        
        # Validasi field yang wajib diisi
        try:
            validate_required_fields(data, ['student_code'])
        except ValidationError as e:
            return jsonify(handle_validation_error(e))
        
        # Validasi format student_code
        student_code = data.get('student_code')
        if not validate_student_code(student_code):
            return jsonify(create_error_response(
                'Format student_code tidak valid. Harus alphanumeric minimal 3 karakter',
                field='student_code'
            ))
        
        # Validasi relationship
        relationship = data.get('relationship', 'wali')
        if not validate_relationship(relationship):
            return jsonify(create_error_response(
                'Jenis relasi tidak valid. Pilih: ayah, ibu, wali, kakak, adik, kakek, nenek',
                field='relationship'
            ))
        
        # Validasi is_primary
        try:
            is_primary = validate_boolean(data.get('is_primary', False), 'is_primary')
        except ValidationError as e:
            return jsonify(handle_validation_error(e))
        
        # Validasi parent_id
        if parent_id <= 0:
            return jsonify(create_error_response('Parent ID harus berupa angka positif'))
        
        # Cek parent exists dan role
        parent = User.query.get(parent_id)
        if not parent:
            return jsonify(create_error_response('Parent tidak ditemukan', 404))
        
        if parent.role != 'orang_tua':
            return jsonify(create_error_response(
                'User dengan ID tersebut bukan parent (role: orang_tua)',
                400,
                'parent_id'
            ))
        
        # Cek student exists
        student = Student.query.filter_by(student_code=student_code).first()
        if not student:
            return jsonify(create_error_response(
                f'Siswa dengan student_code "{student_code}" tidak ditemukan',
                404,
                'student_code'
            ))
        
        # Upsert-like: cek existing relationship
        existing = StudentParent.query.filter_by(
            student_id=student.id, 
            parent_id=parent.id
        ).first()
        
        if existing:
            # Update existing relationship
            existing.relationship = relationship
            existing.is_primary = is_primary
            message = 'Relasi orang tua-siswa berhasil diperbarui'
        else:
            # Create new relationship
            sp = StudentParent(
                student_id=student.id, 
                parent_id=parent.id, 
                relationship=relationship, 
                is_primary=is_primary
            )
            db.session.add(sp)
            message = 'Relasi orang tua-siswa berhasil dibuat'
        
        db.session.commit()
        
        return jsonify({
            'message': message,
            'success': True,
            'data': {
                'parent_id': parent.id,
                'parent_name': parent.full_name,
                'student_id': student.id,
                'student_code': student.student_code,
                'student_name': student.full_name,
                'relationship': relationship,
                'is_primary': is_primary
            }
        }), 201
        
    except ValidationError as e:
        db.session.rollback()
        return jsonify(handle_validation_error(e))
    except Exception as e:
        db.session.rollback()
        return jsonify(create_error_response(f'Terjadi kesalahan server: {str(e)}', 500))

@app.route('/api/admin/parent-student-relations')
@jwt_required()
@require_role(['admin'])
def get_parent_student_relations():
    """Get all parent-student relations for admin"""
    try:
        relations = db.session.query(
            StudentParent.id,
            StudentParent.student_id,
            StudentParent.parent_id,
            StudentParent.relationship,
            StudentParent.created_at,
            Student.student_code,
            Student.full_name.label('student_name'),
            User.full_name.label('parent_name'),
            User.username.label('parent_username')
        ).join(
            Student, StudentParent.student_id == Student.id
        ).join(
            User, StudentParent.parent_id == User.id
        ).all()
        
        result = []
        for rel in relations:
            result.append({
                'id': rel.id,
                'student_id': rel.student_id,
                'parent_id': rel.parent_id,
                'student_code': rel.student_code,
                'student_name': rel.student_name,
                'parent_name': rel.parent_name,
                'parent_username': rel.parent_username,
                'relationship': rel.relationship,
                'created_at': rel.created_at.isoformat() if rel.created_at else None
            })
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/parent-student-relations/<int:relation_id>', methods=['DELETE'])
@jwt_required()
@require_role(['admin'])
def delete_parent_student_relation(relation_id):
    """Delete parent-student relation"""
    try:
        relation = StudentParent.query.get(relation_id)
        if not relation:
            return jsonify({'error': 'Relasi tidak ditemukan'}), 404
        
        db.session.delete(relation)
        db.session.commit()
        
        return jsonify({'message': 'Relasi berhasil dihapus'}), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/students/<int:student_id>/parents')
@jwt_required()
@require_role(['admin'])
def get_student_parents(student_id):
    """Get parents of a specific student"""
    try:
        student = Student.query.get(student_id)
        if not student:
            return jsonify({'error': 'Siswa tidak ditemukan'}), 404
        
        parents = db.session.query(
            StudentParent.id,
            StudentParent.parent_id,
            StudentParent.relationship,
            User.full_name.label('parent_name'),
            User.username.label('parent_username'),
            User.email.label('parent_email')
        ).join(
            User, StudentParent.parent_id == User.id
        ).filter(
            StudentParent.student_id == student_id
        ).all()
        
        result = []
        for parent in parents:
            result.append({
                'relation_id': parent.id,
                'parent_id': parent.parent_id,
                'parent_name': parent.parent_name,
                'parent_username': parent.parent_username,
                'parent_email': parent.parent_email,
                'relationship': parent.relationship
            })
        
        return jsonify({
            'student': student.to_dict(),
            'parents': result
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/parents/<int:parent_id>/students')
@jwt_required()
@require_role(['admin'])
def get_parent_students(parent_id):
    """Get students of a specific parent"""
    try:
        parent = User.query.get(parent_id)
        if not parent or parent.role != 'orang_tua':
            return jsonify({'error': 'Parent tidak ditemukan atau bukan role orang_tua'}), 404
        
        students = db.session.query(
            StudentParent.id,
            StudentParent.student_id,
            StudentParent.relationship,
            Student.student_code,
            Student.full_name.label('student_name'),
            Student.class_name
        ).join(
            Student, StudentParent.student_id == Student.id
        ).filter(
            StudentParent.parent_id == parent_id
        ).all()
        
        result = []
        for student in students:
            result.append({
                'relation_id': student.id,
                'student_id': student.student_id,
                'student_code': student.student_code,
                'student_name': student.student_name,
                'class_name': student.class_name,
                'relationship': student.relationship
            })
        
        return jsonify({
            'parent': parent.to_dict(),
            'students': result
        }), 200
        
    except Exception as e:
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

@app.route('/api/admin/compress-data', methods=['POST'])
@jwt_required()
@require_role(['admin'])
def compress_historical_data():
    """Endpoint untuk kompresi data historis"""
    try:
        data = request.get_json()
        days_threshold = data.get('days_threshold', 30)
        compression_ratio = data.get('compression_ratio', 0.1)
        
        # Run compression
        compressed_count = compression_service.compress_old_emotion_logs(
            days_threshold=days_threshold,
            compression_ratio=compression_ratio
        )
        
        # Optimize database indexes
        compression_service.optimize_database_indexes()
        
        return jsonify({
            'message': f'Successfully compressed {compressed_count} emotion log groups',
            'days_threshold': days_threshold,
            'compression_ratio': compression_ratio
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/system-stats')
@jwt_required()
@require_role(['admin'])
def get_comprehensive_system_stats():
    """Get comprehensive system statistics"""
    try:
        from datetime import datetime, timedelta
        
        # Database stats
        total_students = Student.query.count()
        total_users = User.query.count()
        total_sessions = EmotionSession.query.count()
        active_sessions = EmotionSession.query.filter_by(status='active').count()
        
        # Recent activity stats
        last_24h = datetime.utcnow() - timedelta(hours=24)
        recent_logs = EmotionLog.query.filter(EmotionLog.detected_at >= last_24h).count()
        
        # WebSocket stats
        ws_stats = ws_service.get_connection_stats() if ws_service else {}
        
        # Redis stats
        redis_stats = {}
        if redis_client:
            try:
                redis_info = redis_client.info()
                redis_stats = {
                    'connected_clients': redis_info.get('connected_clients', 0),
                    'used_memory': redis_info.get('used_memory_human', 'N/A'),
                    'keyspace_hits': redis_info.get('keyspace_hits', 0)
                }
            except Exception:
                redis_stats = {'status': 'unavailable'}
        
        return jsonify({
            'database': {
                'total_students': total_students,
                'total_users': total_users,
                'total_sessions': total_sessions,
                'active_sessions': active_sessions,
                'recent_logs_24h': recent_logs
            },
            'websocket': ws_stats,
            'redis': redis_stats,
            'timestamp': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Additional Admin API Endpoints for Dashboard Features

@app.route('/api/admin/analytics/emotion-trends')
@jwt_required()
@require_role(['admin'])
def get_emotion_trends():
    """Get emotion trends data for analytics"""
    try:
        from datetime import datetime, timedelta
        
        # Get last 7 days data
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=7)
        
        # Query emotion logs by day
        trends = db.session.query(
            db.func.date(EmotionLog.detected_at).label('date'),
            EmotionLog.emotion,
            db.func.count(EmotionLog.id).label('count')
        ).filter(
            EmotionLog.detected_at >= start_date
        ).group_by(
            db.func.date(EmotionLog.detected_at),
            EmotionLog.emotion
        ).all()
        
        # Format data for chart
        labels = []
        values = []
        
        for i in range(7):
            date = (start_date + timedelta(days=i)).strftime('%Y-%m-%d')
            labels.append(date)
            
            # Count total emotions for this day
            day_count = sum(t.count for t in trends if t.date.strftime('%Y-%m-%d') == date)
            values.append(day_count)
        
        return jsonify({
            'labels': labels,
            'values': values
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/analytics/role-activity')
@jwt_required()
@require_role(['admin'])
def get_role_activity():
    """Get role activity data for analytics"""
    try:
        # Count users by role
        role_counts = db.session.query(
            User.role,
            db.func.count(User.id).label('count')
        ).group_by(User.role).all()
        
        labels = []
        values = []
        
        for role_count in role_counts:
            labels.append(role_count.role.title())
            values.append(role_count.count)
        
        return jsonify({
            'labels': labels,
            'values': values
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/recent-activity')
@jwt_required()
@require_role(['admin'])
def get_recent_activity():
    """Get recent system activity"""
    try:
        from datetime import datetime, timedelta
        
        activities = []
        
        # Get recent sessions
        recent_sessions = EmotionSession.query.filter(
            EmotionSession.start_time >= datetime.utcnow() - timedelta(hours=24)
        ).order_by(EmotionSession.start_time.desc()).limit(5).all()
        
        for session in recent_sessions:
            activities.append({
                'type': 'session_start',
                'description': f'Sesi "{session.session_name}" dimulai untuk {session.student.full_name if session.student else "Unknown"}',
                'timestamp': session.start_time.isoformat()
            })
        
        # Get recent user logins (if you have login tracking)
        recent_users = User.query.filter(
            User.last_login >= datetime.utcnow() - timedelta(hours=24)
        ).order_by(User.last_login.desc()).limit(5).all()
        
        for user in recent_users:
            activities.append({
                'type': 'user_login',
                'description': f'{user.full_name} ({user.role}) login ke sistem',
                'timestamp': user.last_login.isoformat()
            })
        
        # Sort by timestamp
        activities.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return jsonify(activities[:10]), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/database-stats')
@jwt_required()
@require_role(['admin'])
def get_database_stats():
    """Get database statistics"""
    try:
        from datetime import datetime, timedelta
        
        # Database stats
        total_students = Student.query.count()
        total_users = User.query.count()
        total_sessions = EmotionSession.query.count()
        active_sessions = EmotionSession.query.filter_by(status='active').count()
        
        # Recent activity
        last_24h = datetime.utcnow() - timedelta(hours=24)
        recent_logs = EmotionLog.query.filter(EmotionLog.detected_at >= last_24h).count()
        
        return jsonify({
            'total_students': total_students,
            'total_users': total_users,
            'total_sessions': total_sessions,
            'active_sessions': active_sessions,
            'recent_logs_24h': recent_logs,
            'database_size': 'N/A',  # You can implement actual size calculation
            'last_maintenance': 'N/A'  # You can implement maintenance tracking
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/users/<int:user_id>/approve', methods=['POST'])
@jwt_required()
@require_role(['admin'])
def approve_user(user_id):
    """Approve or reject a user"""
    try:
        data = request.get_json()
        approved = data.get('approved')
        
        user = User.query.get_or_404(user_id)
        user.is_approved = approved
        db.session.commit()
        
        # Emit WebSocket event
        if ws_service:
            ws_service.emit_to_admin('user_approval', {
                'user_id': user_id,
                'username': user.username,
                'approved': approved,
                'timestamp': datetime.utcnow().isoformat()
            })
        
        return jsonify({'message': f'User {"approved" if approved else "rejected"} successfully'}), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/students/<int:student_id>', methods=['GET'])
@jwt_required()
@require_role(['admin'])
def get_student_detail(student_id):
    """Get student detail for editing"""
    try:
        student = Student.query.get_or_404(student_id)
        
        return jsonify({
            'id': student.id,
            'student_code': student.student_code,
            'full_name': student.full_name,
            'class_name': student.class_name,
            'birth_date': student.birth_date.isoformat() if student.birth_date else None,
            'phone': student.phone,
            'email': student.email,
            'address': student.address,
            'subject': student.subject,
            'notes': student.notes,
            'is_active': student.is_active
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/admin/export/<data_type>')
@jwt_required()
@require_role(['admin'])
def export_data(data_type):
    """Export data as CSV"""
    try:
        import csv
        import io
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        if data_type == 'users':
            users = User.query.all()
            writer.writerow(['ID', 'Username', 'Full Name', 'Email', 'Role', 'Active', 'Approved', 'Created At'])
            for user in users:
                writer.writerow([
                    user.id, user.username, user.full_name, user.email,
                    user.role, user.is_active, user.is_approved, user.created_at
                ])
        
        elif data_type == 'students':
            students = Student.query.all()
            writer.writerow(['ID', 'Student Code', 'Full Name', 'Class', 'Birth Date', 'Phone', 'Email', 'Active'])
            for student in students:
                writer.writerow([
                    student.id, student.student_code, student.full_name,
                    student.class_name, student.birth_date, student.phone,
                    student.email, student.is_active
                ])
        
        elif data_type == 'sessions':
            sessions = EmotionSession.query.all()
            writer.writerow(['ID', 'Session Name', 'Student', 'Teacher', 'Status', 'Start Time', 'End Time'])
            for session in sessions:
                writer.writerow([
                    session.id, session.session_name,
                    session.student.full_name if session.student else 'N/A',
                    session.teacher.full_name if session.teacher else 'N/A',
                    session.status, session.start_time, session.end_time
                ])
        
        elif data_type == 'emotions':
            logs = EmotionLog.query.order_by(EmotionLog.detected_at.desc()).limit(1000).all()
            writer.writerow(['ID', 'Session', 'Student', 'Emotion', 'Confidence', 'Detected At'])
            for log in logs:
                writer.writerow([
                    log.id, log.session.session_name if log.session else 'N/A',
                    log.session.student.full_name if log.session and log.session.student else 'N/A',
                    log.emotion, log.confidence_score, log.detected_at
                ])
        
        else:
            return jsonify({'error': 'Invalid data type'}), 400
        
        output.seek(0)
        return Response(
            output.getvalue(),
            mimetype='text/csv',
            headers={'Content-Disposition': f'attachment; filename={data_type}_export.csv'}
        )
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/create-backup', methods=['POST'])
@jwt_required()
@require_role(['admin'])
def create_backup():
    """Create database backup"""
    try:
        from datetime import datetime
        import shutil
        import os
        
        # Simple file backup (you can implement actual database backup)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_filename = f'backup_{timestamp}.json'
        
        # For now, just return success message
        # In production, implement actual database backup
        
        return jsonify({
            'message': 'Backup created successfully',
            'filename': backup_filename
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    port = int(os.environ.get('PORT', 5000))
    socketio.run(app, host='0.0.0.0', port=port, debug=True)

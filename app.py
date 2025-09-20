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


def generate_frames():
    cap = _open_camera_with_fallback()
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
                        db_path=GALLERY_DIR,
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
    global SESSION_START_TS
    SESSION_START_TS = time.time()
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/camera/health')
def camera_health():
    cap = _open_camera_with_fallback()
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
        'cameraAvailable': bool(_open_camera_with_fallback() and _open_camera_with_fallback().isOpened()),
        'status': 'ngrok' if is_ngrok else 'local'
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
        return jsonify({'emotion': emotion})
        
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

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

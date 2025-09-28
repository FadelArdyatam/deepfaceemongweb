"""
Emotion Processing Service untuk optimasi real-time processing
"""
import cv2
import numpy as np
from collections import deque, Counter
from deepface import DeepFace
import threading
import queue
import time
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class EmotionProcessor:
    def __init__(self, max_queue_size=10, processing_interval=0.1):
        self.max_queue_size = max_queue_size
        self.processing_interval = processing_interval
        self.frame_queue = queue.Queue(maxsize=max_queue_size)
        self.emotion_history = deque(maxlen=10)
        self.processing = False
        self.thread = None
        self.callbacks = []
        
    def add_callback(self, callback):
        """Add callback untuk emotion detection"""
        self.callbacks.append(callback)
    
    def start_processing(self):
        """Start background processing thread"""
        if self.processing:
            return
            
        self.processing = True
        self.thread = threading.Thread(target=self._process_frames, daemon=True)
        self.thread.start()
        logger.info("Emotion processor started")
    
    def stop_processing(self):
        """Stop background processing"""
        self.processing = False
        if self.thread:
            self.thread.join(timeout=1)
        logger.info("Emotion processor stopped")
    
    def add_frame(self, frame, metadata=None):
        """Add frame to processing queue"""
        if self.frame_queue.full():
            # Remove oldest frame if queue is full
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                pass
                
        frame_data = {
            'frame': frame.copy(),
            'timestamp': time.time(),
            'metadata': metadata or {}
        }
        
        try:
            self.frame_queue.put_nowait(frame_data)
        except queue.Full:
            logger.warning("Frame queue full, dropping frame")
    
    def _process_frames(self):
        """Background frame processing"""
        while self.processing:
            try:
                # Get frame with timeout
                frame_data = self.frame_queue.get(timeout=self.processing_interval)
                self._process_single_frame(frame_data)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Frame processing error: {e}")
    
    def _process_single_frame(self, frame_data):
        """Process single frame untuk emotion detection"""
        try:
            frame = frame_data['frame']
            timestamp = frame_data['timestamp']
            
            # Skip processing if queue has newer frames
            if not self.frame_queue.empty():
                return
                
            # Intelligent frame skipping based on motion
            if self._should_skip_frame(frame):
                return
                
            # Process emotion
            emotion = self._detect_emotion(frame)
            if emotion:
                self.emotion_history.append(emotion)
                
                # Smooth emotion using history
                smoothed_emotion = self._smooth_emotion()
                
                # Notify callbacks
                for callback in self.callbacks:
                    try:
                        callback({
                            'emotion': smoothed_emotion,
                            'timestamp': timestamp,
                            'metadata': frame_data['metadata']
                        })
                    except Exception as e:
                        logger.error(f"Callback error: {e}")
                        
        except Exception as e:
            logger.error(f"Single frame processing error: {e}")
    
    def _should_skip_frame(self, frame):
        """Determine if frame should be skipped based on motion"""
        # Simple motion detection
        if len(self.emotion_history) < 3:
            return False
            
        # Skip if emotion hasn't changed much
        recent_emotions = list(self.emotion_history)[-3:]
        if len(set(recent_emotions)) == 1:  # All same emotion
            return True
            
        return False
    
    def _detect_emotion(self, frame):
        """Detect emotion from frame"""
        try:
            # Resize frame untuk performa
            height, width = frame.shape[:2]
            if width > 640:
                scale = 640 / width
                new_width = int(width * scale)
                new_height = int(height * scale)
                frame = cv2.resize(frame, (new_width, new_height))
            
            # Process dengan DeepFace
            result = DeepFace.analyze(
                frame,
                actions=['emotion'],
                detector_backend='opencv',
                enforce_detection=False,
                silent=True
            )
            
            if result and len(result) > 0:
                return result[0]['dominant_emotion']
                
        except Exception as e:
            logger.warning(f"Emotion detection error: {e}")
            
        return None
    
    def _smooth_emotion(self):
        """Smooth emotion using history"""
        if not self.emotion_history:
            return "neutral"
            
        # Use most common emotion from recent history
        emotion_counts = Counter(self.emotion_history)
        return emotion_counts.most_common(1)[0][0]

class EmotionAggregator:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.emotion_windows = {}
        
    def add_emotion(self, student_id, emotion, timestamp):
        """Add emotion to student's window"""
        if student_id not in self.emotion_windows:
            self.emotion_windows[student_id] = deque(maxlen=self.window_size)
            
        self.emotion_windows[student_id].append({
            'emotion': emotion,
            'timestamp': timestamp
        })
    
    def get_dominant_emotion(self, student_id):
        """Get dominant emotion for student"""
        if student_id not in self.emotion_windows:
            return None
            
        window = self.emotion_windows[student_id]
        if not window:
            return None
            
        emotions = [entry['emotion'] for entry in window]
        emotion_counts = Counter(emotions)
        return emotion_counts.most_common(1)[0][0]
    
    def get_emotion_trend(self, student_id):
        """Get emotion trend for student"""
        if student_id not in self.emotion_windows:
            return []
            
        window = self.emotion_windows[student_id]
        return [entry['emotion'] for entry in window]

# Global instances
emotion_processor = EmotionProcessor()
emotion_aggregator = EmotionAggregator()
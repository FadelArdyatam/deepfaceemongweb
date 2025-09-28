"""
WebSocket Service untuk optimasi real-time communication
"""
from flask_socketio import SocketIO, emit, join_room, leave_room
import redis
import json
import logging
from datetime import datetime, timedelta
from collections import defaultdict
import threading
import time

logger = logging.getLogger(__name__)

class WebSocketService:
    def __init__(self, socketio, redis_client=None):
        self.socketio = socketio
        self.redis = redis_client
        self.connection_pool = {}
        self.message_queue = defaultdict(list)
        self.room_subscriptions = defaultdict(set)
        self.heartbeat_interval = 30  # seconds
        self.heartbeat_thread = None
        
    def start_heartbeat(self):
        """Start heartbeat monitoring"""
        if self.heartbeat_thread and self.heartbeat_thread.is_alive():
            return
            
        self.heartbeat_thread = threading.Thread(target=self._heartbeat_monitor, daemon=True)
        self.heartbeat_thread.start()
        logger.info("WebSocket heartbeat started")
    
    def _heartbeat_monitor(self):
        """Monitor connection health"""
        while True:
            try:
                time.sleep(self.heartbeat_interval)
                self._cleanup_stale_connections()
            except Exception as e:
                logger.error(f"Heartbeat monitor error: {e}")
    
    def _cleanup_stale_connections(self):
        """Clean up stale connections"""
        current_time = time.time()
        stale_connections = []
        
        for connection_id, last_seen in self.connection_pool.items():
            if current_time - last_seen > self.heartbeat_interval * 2:
                stale_connections.append(connection_id)
        
        for connection_id in stale_connections:
            self._remove_connection(connection_id)
    
    def _remove_connection(self, connection_id):
        """Remove connection from pool"""
        if connection_id in self.connection_pool:
            del self.connection_pool[connection_id]
            
        # Remove from room subscriptions
        for room, connections in self.room_subscriptions.items():
            connections.discard(connection_id)
    
    def register_connection(self, connection_id, user_id, role):
        """Register new connection"""
        self.connection_pool[connection_id] = {
            'user_id': user_id,
            'role': role,
            'last_seen': time.time(),
            'rooms': set()
        }
        logger.info(f"Connection registered: {connection_id} ({role})")
    
    def join_user_room(self, connection_id, user_id, role):
        """Join user to appropriate room"""
        if connection_id not in self.connection_pool:
            return False
            
        room_name = f"{role}:{user_id}"
        join_room(room_name)
        
        self.connection_pool[connection_id]['rooms'].add(room_name)
        self.room_subscriptions[room_name].add(connection_id)
        
        logger.info(f"User {user_id} joined room {room_name}")
        return True
    
    def emit_to_user(self, user_id, role, event, data):
        """Emit event to specific user"""
        room_name = f"{role}:{user_id}"
        self.socketio.emit(event, data, to=room_name)
        logger.debug(f"Emitted {event} to {room_name}")
    
    def emit_to_role(self, role, event, data):
        """Emit event to all users with specific role"""
        for connection_id, info in self.connection_pool.items():
            if info['role'] == role:
                self.emit_to_user(info['user_id'], role, event, data)
    
    def broadcast_emotion_update(self, student_id, emotion, timestamp):
        """Broadcast emotion update to relevant parents"""
        try:
            # Get parents of student from cache or DB
            parent_ids = self._get_student_parents(student_id)
            
            for parent_id in parent_ids:
                self.emit_to_user(parent_id, 'orang_tua', 'emotion_log_created', {
                    'student_id': student_id,
                    'emotion': emotion,
                    'detected_at': timestamp
                })
                
        except Exception as e:
            logger.error(f"Broadcast emotion error: {e}")
    
    def broadcast_session_update(self, teacher_id, session_data):
        """Broadcast session update to teacher"""
        self.emit_to_user(teacher_id, 'guru', 'session_update', session_data)
    
    def broadcast_system_stats(self, stats_data):
        """Broadcast system stats to all admins"""
        self.emit_to_role('admin', 'system_stats', stats_data)
    
    def queue_offline_message(self, user_id, role, event, data):
        """Queue message for offline user"""
        if not self.redis:
            return
            
        try:
            message_key = f"offline_messages:{role}:{user_id}"
            message = {
                'event': event,
                'data': data,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Add to queue with expiration
            self.redis.lpush(message_key, json.dumps(message))
            self.redis.expire(message_key, 3600)  # 1 hour expiration
            
        except Exception as e:
            logger.error(f"Queue offline message error: {e}")
    
    def deliver_offline_messages(self, user_id, role):
        """Deliver queued messages to user"""
        if not self.redis:
            return
            
        try:
            message_key = f"offline_messages:{role}:{user_id}"
            messages = self.redis.lrange(message_key, 0, -1)
            
            for message_json in messages:
                message = json.loads(message_json)
                self.emit_to_user(user_id, role, message['event'], message['data'])
            
            # Clear delivered messages
            self.redis.delete(message_key)
            
        except Exception as e:
            logger.error(f"Deliver offline messages error: {e}")
    
    def _get_student_parents(self, student_id):
        """Get parent IDs for student (cached)"""
        if not self.redis:
            return []
            
        try:
            cache_key = f"student_parents:{student_id}"
            cached = self.redis.get(cache_key)
            
            if cached:
                return json.loads(cached)
                
            # Query database if not cached
            from models import StudentParent
            parent_rows = StudentParent.query.filter_by(student_id=student_id).all()
            parent_ids = [row.parent_id for row in parent_rows]
            
            # Cache for 5 minutes
            self.redis.setex(cache_key, 300, json.dumps(parent_ids))
            return parent_ids
            
        except Exception as e:
            logger.error(f"Get student parents error: {e}")
            return []
    
    def update_connection_heartbeat(self, connection_id):
        """Update connection heartbeat"""
        if connection_id in self.connection_pool:
            self.connection_pool[connection_id]['last_seen'] = time.time()
    
    def get_connection_stats(self):
        """Get connection statistics"""
        stats = {
            'total_connections': len(self.connection_pool),
            'connections_by_role': defaultdict(int),
            'room_subscriptions': len(self.room_subscriptions)
        }
        
        for connection_id, info in self.connection_pool.items():
            stats['connections_by_role'][info['role']] += 1
            
        return stats

# Global WebSocket service instance
websocket_service = None

def init_websocket_service(socketio, redis_client=None):
    """Initialize global WebSocket service"""
    global websocket_service
    websocket_service = WebSocketService(socketio, redis_client)
    websocket_service.start_heartbeat()
    return websocket_service
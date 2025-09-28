"""
Optimized API Routes dengan caching dan bulk operations
"""
from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from services.database_service import DatabaseService
from services.websocket_service import websocket_service
from functools import wraps
import logging

logger = logging.getLogger(__name__)

# Create blueprint
api_bp = Blueprint('optimized_api', __name__, url_prefix='/api/v2')

# Initialize services
db_service = None
ws_service = None

def init_services(db, redis_client, socketio):
    """Initialize services"""
    global db_service, ws_service
    db_service = DatabaseService(db, redis_client)
    ws_service = websocket_service

def require_role(roles):
    """Role-based access control decorator"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            try:
                user_id = get_jwt_identity()
                user = db_service.get_user_with_relations(user_id)
                if not user or user['user']['role'] not in roles:
                    return jsonify({'error': 'Access denied'}), 403
                return f(*args, **kwargs)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        return decorated_function
    return decorator

@api_bp.route('/dashboard/stats')
@jwt_required()
def get_optimized_dashboard_stats():
    """Optimized dashboard stats dengan caching"""
    try:
        user_id = get_jwt_identity()
        user_data = db_service.get_user_with_relations(user_id)
        
        if not user_data:
            return jsonify({'error': 'User not found'}), 404
            
        user_role = user_data['user']['role']
        stats = db_service.get_dashboard_stats_optimized(user_id, user_role)
        
        # Add real-time data
        if ws_service:
            connection_stats = ws_service.get_connection_stats()
            stats['real_time'] = connection_stats
            
        return jsonify(stats), 200
        
    except Exception as e:
        logger.error(f"Dashboard stats error: {e}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/students/bulk', methods=['POST'])
@jwt_required()
@require_role(['guru', 'admin'])
def bulk_create_students():
    """Bulk create students untuk efisiensi"""
    try:
        data = request.get_json()
        students_data = data.get('students', [])
        
        if not students_data:
            return jsonify({'error': 'No students data provided'}), 400
            
        # Validate all students first
        for student_data in students_data:
            required_fields = ['student_code', 'full_name', 'class_name']
            for field in required_fields:
                if field not in student_data or not student_data[field]:
                    return jsonify({'error': f'Field {field} is required for all students'}), 400
        
        # Bulk create
        from models import Student, StudentTeacher
        user_id = get_jwt_identity()
        
        created_students = []
        for student_data in students_data:
            student = Student(
                student_code=student_data['student_code'],
                full_name=student_data['full_name'],
                class_name=student_data['class_name'],
                birth_date=student_data.get('birth_date'),
                photo_path=student_data.get('photo_path')
            )
            db_service.db.session.add(student)
            db_service.db.session.flush()  # Get ID without commit
            
            # Create teacher-student relationship
            if user_id:
                relation = StudentTeacher(
                    teacher_id=user_id,
                    student_id=student.id
                )
                db_service.db.session.add(relation)
            
            created_students.append(student.to_dict())
        
        db_service.db.session.commit()
        
        # Clear cache
        if db_service.redis:
            pattern = f"user:{user_id}:*"
            keys = db_service.redis.keys(pattern)
            if keys:
                db_service.redis.delete(*keys)
        
        return jsonify({
            'message': f'Successfully created {len(created_students)} students',
            'students': created_students
        }), 201
        
    except Exception as e:
        db_service.db.session.rollback()
        logger.error(f"Bulk create students error: {e}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/emotions/bulk', methods=['POST'])
@jwt_required()
@require_role(['guru', 'admin'])
def bulk_create_emotion_logs():
    """Bulk create emotion logs untuk performa"""
    try:
        data = request.get_json()
        logs_data = data.get('logs', [])
        
        if not logs_data:
            return jsonify({'error': 'No logs data provided'}), 400
        
        # Validate logs
        for log_data in logs_data:
            required_fields = ['session_id', 'student_id', 'emotion']
            for field in required_fields:
                if field not in log_data:
                    return jsonify({'error': f'Field {field} is required for all logs'}), 400
        
        # Bulk insert
        success = db_service.bulk_create_emotion_logs(logs_data)
        
        if success:
            # Broadcast updates via WebSocket
            if ws_service:
                for log_data in logs_data:
                    ws_service.broadcast_emotion_update(
                        log_data['student_id'],
                        log_data['emotion'],
                        log_data.get('detected_at', '')
                    )
            
            return jsonify({
                'message': f'Successfully created {len(logs_data)} emotion logs'
            }), 201
        else:
            return jsonify({'error': 'Failed to create emotion logs'}), 500
            
    except Exception as e:
        logger.error(f"Bulk create emotion logs error: {e}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/analytics/emotions')
@jwt_required()
@require_role(['guru', 'admin'])
def get_emotion_analytics():
    """Optimized emotion analytics dengan caching"""
    try:
        user_id = get_jwt_identity()
        days = request.args.get('days', 7, type=int)
        
        analytics = db_service.get_emotion_analytics_optimized(user_id, days)
        
        return jsonify(analytics), 200
        
    except Exception as e:
        logger.error(f"Emotion analytics error: {e}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/system/health')
@jwt_required()
@require_role(['admin'])
def get_system_health():
    """System health check dengan real-time stats"""
    try:
        health_data = {
            'database': 'healthy',
            'redis': 'healthy' if db_service.redis else 'unavailable',
            'websocket': 'healthy' if ws_service else 'unavailable',
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Add connection stats
        if ws_service:
            health_data['connections'] = ws_service.get_connection_stats()
        
        # Add cache stats
        if db_service.redis:
            try:
                cache_info = db_service.redis.info('memory')
                health_data['cache'] = {
                    'used_memory': cache_info.get('used_memory_human'),
                    'connected_clients': cache_info.get('connected_clients')
                }
            except Exception:
                health_data['cache'] = 'unavailable'
        
        return jsonify(health_data), 200
        
    except Exception as e:
        logger.error(f"System health error: {e}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/cache/clear')
@jwt_required()
@require_role(['admin'])
def clear_cache():
    """Clear all cache"""
    try:
        if db_service.redis:
            db_service.redis.flushdb()
            return jsonify({'message': 'Cache cleared successfully'}), 200
        else:
            return jsonify({'error': 'Redis not available'}), 503
            
    except Exception as e:
        logger.error(f"Clear cache error: {e}")
        return jsonify({'error': str(e)}), 500
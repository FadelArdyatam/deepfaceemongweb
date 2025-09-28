"""
Database Service Layer untuk optimasi CRUD operations
"""
from flask_sqlalchemy import SQLAlchemy
from functools import wraps
import redis
import json
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class DatabaseService:
    def __init__(self, db, redis_client=None):
        self.db = db
        self.redis = redis_client
        self.cache_ttl = 300  # 5 minutes cache
        
    def cache_key(self, prefix, *args):
        """Generate cache key"""
        return f"{prefix}:{':'.join(map(str, args))}"
    
    def get_cached(self, key):
        """Get from cache"""
        if not self.redis:
            return None
        try:
            cached = self.redis.get(key)
            return json.loads(cached) if cached else None
        except Exception as e:
            logger.warning(f"Cache get error: {e}")
            return None
    
    def set_cache(self, key, data, ttl=None):
        """Set cache"""
        if not self.redis:
            return
        try:
            ttl = ttl or self.cache_ttl
            self.redis.setex(key, ttl, json.dumps(data, default=str))
        except Exception as e:
            logger.warning(f"Cache set error: {e}")
    
    def get_user_with_relations(self, user_id):
        """Get user with all relations in one query"""
        cache_key = self.cache_key("user", user_id)
        cached = self.get_cached(cache_key)
        if cached:
            return cached
            
        from models import User, Student, EmotionSession
        user = self.db.session.query(User).filter(User.id == user_id).first()
        if not user:
            return None
            
        # Get related data in one query
        if user.role == 'guru':
            students = self.db.session.query(Student).join(
                Student.teachers
            ).filter(Student.teachers.any(teacher_id=user_id)).all()
            
            active_sessions = self.db.session.query(EmotionSession).filter(
                EmotionSession.teacher_id == user_id,
                EmotionSession.status == 'active'
            ).all()
            
            result = {
                'user': user.to_dict(),
                'students': [s.to_dict() for s in students],
                'active_sessions': [s.to_dict() for s in active_sessions]
            }
        else:
            result = {'user': user.to_dict()}
            
        self.set_cache(cache_key, result)
        return result
    
    def get_dashboard_stats_optimized(self, user_id, user_role):
        """Optimized dashboard stats with caching"""
        cache_key = self.cache_key("dashboard_stats", user_id, user_role)
        cached = self.get_cached(cache_key)
        if cached:
            return cached
            
        from models import User, Student, EmotionSession, EmotionLog
        from datetime import date
        
        stats = {}
        
        if user_role == 'guru':
            # Single query untuk semua stats guru
            result = self.db.session.query(
                self.db.func.count(Student.id).label('total_students'),
                self.db.func.count(EmotionSession.id).label('active_sessions'),
                self.db.func.count(EmotionLog.id).label('today_detections')
            ).outerjoin(
                Student.teachers
            ).outerjoin(
                EmotionSession, EmotionSession.teacher_id == user_id
            ).outerjoin(
                EmotionLog, 
                (EmotionLog.session_id == EmotionSession.id) & 
                (self.db.func.date(EmotionLog.detected_at) == date.today())
            ).filter(
                Student.teachers.any(teacher_id=user_id)
            ).first()
            
            stats = {
                'total_students': result.total_students or 0,
                'active_sessions': result.active_sessions or 0,
                'today_detections': result.today_detections or 0
            }
            
        elif user_role == 'admin':
            # Admin stats dengan single query
            result = self.db.session.query(
                self.db.func.count(User.id).label('total_users'),
                self.db.func.count(Student.id).label('total_students'),
                self.db.func.count(EmotionSession.id).label('active_sessions')
            ).outerjoin(Student).outerjoin(EmotionSession).first()
            
            stats = {
                'total_users': result.total_users or 0,
                'total_students': result.total_students or 0,
                'active_sessions': result.active_sessions or 0
            }
            
        self.set_cache(cache_key, stats, 60)  # 1 minute cache
        return stats
    
    def bulk_create_emotion_logs(self, logs_data):
        """Bulk insert emotion logs untuk performa"""
        try:
            from models import EmotionLog
            logs = [EmotionLog(**log_data) for log_data in logs_data]
            self.db.session.bulk_save_objects(logs)
            self.db.session.commit()
            return True
        except Exception as e:
            self.db.session.rollback()
            logger.error(f"Bulk insert error: {e}")
            return False
    
    def get_emotion_analytics_optimized(self, teacher_id, days=7):
        """Optimized emotion analytics dengan single query"""
        cache_key = self.cache_key("emotion_analytics", teacher_id, days)
        cached = self.get_cached(cache_key)
        if cached:
            return cached
            
        from models import EmotionLog, EmotionSession
        from datetime import datetime, timedelta
        
        start_date = datetime.now() - timedelta(days=days)
        
        # Single query untuk analytics
        result = self.db.session.query(
            EmotionLog.emotion,
            self.db.func.count(EmotionLog.id).label('count'),
            self.db.func.date(EmotionLog.detected_at).label('date')
        ).join(
            EmotionSession, EmotionLog.session_id == EmotionSession.id
        ).filter(
            EmotionSession.teacher_id == teacher_id,
            EmotionLog.detected_at >= start_date
        ).group_by(
            EmotionLog.emotion, self.db.func.date(EmotionLog.detected_at)
        ).all()
        
        analytics = {}
        for row in result:
            emotion = row.emotion
            date_str = row.date.isoformat()
            if emotion not in analytics:
                analytics[emotion] = {}
            analytics[emotion][date_str] = row.count
            
        self.set_cache(cache_key, analytics, 300)  # 5 minutes cache
        return analytics

# Decorator untuk caching
def cached_result(ttl=300):
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            cached = self.get_cached(cache_key)
            if cached:
                return cached
            result = func(self, *args, **kwargs)
            self.set_cache(cache_key, result, ttl)
            return result
        return wrapper
    return decorator
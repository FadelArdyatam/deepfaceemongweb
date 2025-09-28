"""
Production Configuration untuk optimasi performa
"""
import os
from config import Config

class ProductionConfig(Config):
    """Production configuration dengan optimasi"""
    
    # Database optimizations
    SQLALCHEMY_ENGINE_OPTIONS = {
        'pool_size': 20,
        'pool_recycle': 3600,
        'pool_pre_ping': True,
        'max_overflow': 30
    }
    
    # Redis configuration
    REDIS_URL = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
    
    # WebSocket optimizations
    SOCKETIO_ASYNC_MODE = 'threading'
    SOCKETIO_PING_TIMEOUT = 60
    SOCKETIO_PING_INTERVAL = 25
    
    # Emotion processing optimizations
    EMOTION_PROCESSING_INTERVAL = 0.1  # seconds
    EMOTION_FRAME_SKIP = 5  # Process every 5th frame
    EMOTION_HISTORY_SIZE = 10
    
    # Caching configuration
    CACHE_DEFAULT_TTL = 300  # 5 minutes
    CACHE_DASHBOARD_TTL = 60  # 1 minute
    CACHE_ANALYTICS_TTL = 300  # 5 minutes
    
    # Performance monitoring
    ENABLE_PERFORMANCE_MONITORING = True
    LOG_LEVEL = 'INFO'
    
    # Security
    JWT_ACCESS_TOKEN_EXPIRES = 3600  # 1 hour
    JWT_REFRESH_TOKEN_EXPIRES = 86400  # 24 hours
    
    # File upload limits
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
    UPLOAD_FOLDER = 'uploads'
    
    # Background tasks
    ENABLE_BACKGROUND_TASKS = True
    BACKGROUND_TASK_INTERVAL = 30  # seconds
"""
Data Compression Service untuk optimasi penyimpanan data historis
"""
import logging
from datetime import datetime, timedelta
from collections import defaultdict
from sqlalchemy import func
import json

logger = logging.getLogger(__name__)

class DataCompressionService:
    def __init__(self, db):
        self.db = db
        
    def compress_old_emotion_logs(self, days_threshold=30, compression_ratio=0.1):
        """
        Compress emotion logs older than threshold days
        Keep only compression_ratio of original data
        """
        try:
            from models import EmotionLog, EmotionSession
            
            cutoff_date = datetime.utcnow() - timedelta(days=days_threshold)
            
            # Get old logs grouped by student and date
            old_logs = self.db.session.query(
                EmotionLog.student_id,
                func.date(EmotionLog.detected_at).label('date'),
                EmotionLog.emotion,
                func.count(EmotionLog.id).label('count'),
                func.avg(EmotionLog.confidence_score).label('avg_confidence'),
                func.min(EmotionLog.detected_at).label('first_detection'),
                func.max(EmotionLog.detected_at).label('last_detection')
            ).filter(
                EmotionLog.detected_at < cutoff_date
            ).group_by(
                EmotionLog.student_id,
                func.date(EmotionLog.detected_at),
                EmotionLog.emotion
            ).all()
            
            # Create compressed records
            compressed_data = []
            for log in old_logs:
                compressed_data.append({
                    'student_id': log.student_id,
                    'date': log.date,
                    'emotion': log.emotion,
                    'total_count': log.count,
                    'avg_confidence': float(log.avg_confidence) if log.avg_confidence else 0.0,
                    'first_detection': log.first_detection.isoformat(),
                    'last_detection': log.last_detection.isoformat(),
                    'compression_ratio': compression_ratio,
                    'compressed_at': datetime.utcnow().isoformat()
                })
            
            # Store compressed data in a separate table or JSON field
            self._store_compressed_data(compressed_data)
            
            # Delete original logs (keep only sample)
            self._delete_old_logs_with_sample(cutoff_date, compression_ratio)
            
            logger.info(f"Compressed {len(compressed_data)} emotion log groups")
            return len(compressed_data)
            
        except Exception as e:
            logger.error(f"Data compression error: {e}")
            return 0
    
    def _store_compressed_data(self, compressed_data):
        """Store compressed data in database"""
        try:
            # Create a simple compressed data table if it doesn't exist
            # For now, we'll store as JSON in a text field
            from models import EmotionAggregation
            
            for data in compressed_data:
                # Store in EmotionAggregation table with special format
                compressed_record = EmotionAggregation(
                    teacher_id=0,  # Special marker for compressed data
                    date=data['date'],
                    emotion=f"COMPRESSED_{data['emotion']}",
                    count=data['total_count']
                )
                self.db.session.add(compressed_record)
            
            self.db.session.commit()
            
        except Exception as e:
            logger.error(f"Store compressed data error: {e}")
            self.db.session.rollback()
    
    def _delete_old_logs_with_sample(self, cutoff_date, keep_ratio):
        """Delete old logs but keep a sample"""
        try:
            from models import EmotionLog
            
            # Get all old logs
            old_logs = EmotionLog.query.filter(
                EmotionLog.detected_at < cutoff_date
            ).order_by(EmotionLog.detected_at).all()
            
            # Keep every nth log based on ratio
            keep_every = int(1 / keep_ratio)
            logs_to_delete = []
            
            for i, log in enumerate(old_logs):
                if i % keep_every != 0:  # Keep every nth log
                    logs_to_delete.append(log.id)
            
            # Delete logs in batches
            batch_size = 1000
            for i in range(0, len(logs_to_delete), batch_size):
                batch = logs_to_delete[i:i + batch_size]
                EmotionLog.query.filter(EmotionLog.id.in_(batch)).delete(synchronize_session=False)
            
            self.db.session.commit()
            logger.info(f"Deleted {len(logs_to_delete)} old emotion logs")
            
        except Exception as e:
            logger.error(f"Delete old logs error: {e}")
            self.db.session.rollback()
    
    def get_compressed_emotion_stats(self, student_id, days=30):
        """Get emotion statistics from compressed data"""
        try:
            from models import EmotionLog, EmotionAggregation
            from datetime import datetime, timedelta
            
            start_date = datetime.utcnow() - timedelta(days=days)
            
            # Get recent uncompressed data
            recent_logs = self.db.session.query(
                EmotionLog.emotion,
                func.count(EmotionLog.id).label('count'),
                func.avg(EmotionLog.confidence_score).label('avg_confidence')
            ).join(EmotionSession).filter(
                EmotionSession.student_id == student_id,
                EmotionLog.detected_at >= start_date
            ).group_by(EmotionLog.emotion).all()
            
            # Get compressed historical data
            compressed_logs = self.db.session.query(
                EmotionAggregation.emotion,
                func.sum(EmotionAggregation.count).label('total_count')
            ).filter(
                EmotionAggregation.teacher_id == 0,  # Compressed data marker
                EmotionAggregation.date >= start_date.date()
            ).group_by(EmotionAggregation.emotion).all()
            
            # Combine data
            stats = {}
            
            # Add recent data
            for log in recent_logs:
                emotion = log.emotion
                stats[emotion] = {
                    'count': log.count,
                    'avg_confidence': float(log.avg_confidence) if log.avg_confidence else 0.0,
                    'source': 'recent'
                }
            
            # Add compressed data
            for log in compressed_logs:
                emotion = log.emotion.replace('COMPRESSED_', '')
                if emotion in stats:
                    stats[emotion]['count'] += log.total_count
                    stats[emotion]['source'] = 'mixed'
                else:
                    stats[emotion] = {
                        'count': log.total_count,
                        'avg_confidence': 0.0,
                        'source': 'compressed'
                    }
            
            return stats
            
        except Exception as e:
            logger.error(f"Get compressed stats error: {e}")
            return {}
    
    def optimize_database_indexes(self):
        """Create optimized indexes for better query performance"""
        try:
            # These would be migration commands in a real implementation
            indexes_to_create = [
                "CREATE INDEX IF NOT EXISTS idx_emotion_logs_student_date ON emotion_logs(student_id, detected_at)",
                "CREATE INDEX IF NOT EXISTS idx_emotion_logs_session_emotion ON emotion_logs(session_id, emotion)",
                "CREATE INDEX IF NOT EXISTS idx_emotion_sessions_student_status ON emotion_sessions(student_id, status)",
                "CREATE INDEX IF NOT EXISTS idx_student_parents_parent ON student_parents(parent_id)",
            ]
            
            for index_sql in indexes_to_create:
                try:
                    self.db.session.execute(index_sql)
                except Exception as e:
                    logger.warning(f"Index creation warning: {e}")
            
            self.db.session.commit()
            logger.info("Database indexes optimized")
            
        except Exception as e:
            logger.error(f"Index optimization error: {e}")
            self.db.session.rollback()

# Global instance
data_compression_service = None

def init_data_compression_service(db):
    """Initialize global data compression service"""
    global data_compression_service
    data_compression_service = DataCompressionService(db)
    return data_compression_service
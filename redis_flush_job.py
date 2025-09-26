#!/usr/bin/env python3
"""
Job periodik untuk auto-flush agregasi Redis ke database
"""

import os
import time
import schedule
import threading
from datetime import datetime, date
from config import Config
import pymysql

# Redis connection
REDIS_URL = os.environ.get('REDIS_URL', '').strip()
redis_client = None
if REDIS_URL:
    try:
        import redis
        redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
        redis_client.ping()
        print("✅ Redis connected for flush job")
    except Exception as e:
        print(f"⚠️  Redis unavailable for flush job: {e}")
        redis_client = None

def flush_redis_to_db():
    """Flush agregasi Redis ke database"""
    if not redis_client:
        print("⚠️  Redis tidak tersedia, skip flush")
        return
    
    try:
        # Koneksi ke database
        connection = pymysql.connect(
            host=Config.DB_HOST,
            user=Config.DB_USERNAME,
            password=Config.DB_PASSWORD,
            database=Config.DB_NAME,
            charset='utf8mb4'
        )
        
        with connection.cursor() as cursor:
            # Ambil semua key agregasi dari Redis
            pattern = "emagg:*"
            keys = redis_client.keys(pattern)
            
            if not keys:
                print("📊 Tidak ada data agregasi untuk di-flush")
                return
            
            flushed_count = 0
            for key in keys:
                try:
                    # Parse key: emagg:teacher_id:date
                    parts = key.split(':')
                    if len(parts) != 3:
                        continue
                    
                    teacher_id = int(parts[1])
                    date_str = parts[2]
                    
                    # Ambil semua emotion counts dari hash
                    emotion_counts = redis_client.hgetall(key)
                    
                    for emotion, count in emotion_counts.items():
                        count = int(count)
                        
                        # Upsert ke database
                        upsert_sql = """
                        INSERT INTO emotion_aggregations (teacher_id, date, emotion, count, created_at, updated_at)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        ON DUPLICATE KEY UPDATE
                        count = count + %s,
                        updated_at = %s
                        """
                        
                        now = datetime.utcnow()
                        cursor.execute(upsert_sql, (
                            teacher_id, date_str, emotion, count, now, now,
                            count, now
                        ))
                        flushed_count += 1
                    
                    # Hapus key dari Redis setelah di-flush
                    redis_client.delete(key)
                    
                except Exception as e:
                    print(f"❌ Error processing key {key}: {e}")
                    continue
            
            connection.commit()
            print(f"✅ Flushed {flushed_count} agregasi ke database")
            
    except Exception as e:
        print(f"❌ Error in flush job: {e}")
    finally:
        if 'connection' in locals():
            connection.close()

def run_flush_job():
    """Jalankan job flush dalam thread terpisah"""
    print("🔄 Starting Redis flush job...")
    flush_redis_to_db()

def start_scheduler():
    """Mulai scheduler untuk job periodik"""
    if not redis_client:
        print("⚠️  Redis tidak tersedia, scheduler tidak dijalankan")
        return
    
    # Schedule job setiap 5 menit
    schedule.every(5).minutes.do(run_flush_job)
    
    # Schedule job setiap hari jam 23:59 untuk flush final
    schedule.every().day.at("23:59").do(run_flush_job)
    
    print("⏰ Scheduler Redis flush job dimulai")
    print("   - Setiap 5 menit")
    print("   - Setiap hari jam 23:59")
    
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check setiap menit

def start_flush_job_background():
    """Jalankan flush job di background thread"""
    def run_scheduler():
        start_scheduler()
    
    scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
    scheduler_thread.start()
    print("🚀 Redis flush job background thread dimulai")
    return scheduler_thread

if __name__ == "__main__":
    # Test flush sekali
    print("🧪 Testing flush job...")
    flush_redis_to_db()
    
    # Jalankan scheduler
    print("\n🔄 Starting scheduler...")
    start_scheduler()
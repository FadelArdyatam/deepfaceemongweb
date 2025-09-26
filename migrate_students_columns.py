#!/usr/bin/env python3
"""
Migration helper: ensure required columns exist on `students` table.
Adds columns if missing: address (TEXT), phone (VARCHAR(20)), email (VARCHAR(120)),
subject (VARCHAR(100)), is_active (BOOLEAN DEFAULT TRUE).

Designed for MySQL (PyMySQL) and uses Config for DB credentials.
"""

import pymysql
from config import Config


REQUIRED_COLUMNS = {
    'address': "TEXT",
    'phone': "VARCHAR(20)",
    'email': "VARCHAR(120)",
    'subject': "VARCHAR(100)",
    'is_active': "TINYINT(1) DEFAULT 1"
}


def column_exists(cursor, table_name: str, column_name: str) -> bool:
    cursor.execute(
        """
        SELECT COUNT(*) FROM information_schema.COLUMNS
        WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s AND COLUMN_NAME = %s
        """,
        (Config.DB_NAME, table_name, column_name)
    )
    count = cursor.fetchone()[0]
    return count > 0


def add_missing_columns():
    connection = None
    try:
        connection = pymysql.connect(
            host=Config.DB_HOST,
            user=Config.DB_USERNAME,
            password=Config.DB_PASSWORD,
            database=Config.DB_NAME,
            charset='utf8mb4'
        )

        with connection.cursor() as cursor:
            for col, col_type in REQUIRED_COLUMNS.items():
                if not column_exists(cursor, 'students', col):
                    print(f"🔄 Adding missing column: {col} ({col_type}) ...")
                    cursor.execute(f"ALTER TABLE students ADD COLUMN {col} {col_type}")
                    connection.commit()
                    print(f"✅ Column added: {col}")
                else:
                    print(f"✅ Column already exists: {col}")

        print("🎉 Migration completed successfully.")
        return True

    except Exception as e:
        print(f"❌ Migration failed: {e}")
        return False

    finally:
        if connection is not None:
            connection.close()


if __name__ == '__main__':
    add_missing_columns()


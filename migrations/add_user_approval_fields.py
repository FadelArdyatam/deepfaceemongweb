#!/usr/bin/env python3
"""
Migration script to add is_approved and last_login fields to User model
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app, db
from models import User

def add_user_approval_fields():
    """Add is_approved and last_login fields to users table"""
    with app.app_context():
        try:
            # Add is_approved column
            with db.engine.connect() as conn:
                conn.execute(db.text("""
                    ALTER TABLE users 
                    ADD COLUMN is_approved BOOLEAN DEFAULT NULL
                """))
                conn.commit()
            print("✅ Added is_approved column to users table")
            
            # Add last_login column
            with db.engine.connect() as conn:
                conn.execute(db.text("""
                    ALTER TABLE users 
                    ADD COLUMN last_login DATETIME DEFAULT NULL
                """))
                conn.commit()
            print("✅ Added last_login column to users table")
            
            # Set existing admin users as approved
            admin_users = User.query.filter_by(role='admin').all()
            for user in admin_users:
                user.is_approved = True
            db.session.commit()
            print(f"✅ Set {len(admin_users)} admin users as approved")
            
            print("🎉 Migration completed successfully!")
            
        except Exception as e:
            print(f"❌ Migration failed: {str(e)}")
            db.session.rollback()
            raise

if __name__ == '__main__':
    add_user_approval_fields()
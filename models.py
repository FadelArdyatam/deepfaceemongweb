from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()

class User(db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    full_name = db.Column(db.String(100), nullable=False)
    role = db.Column(db.Enum('admin', 'guru', 'orang_tua'), nullable=False)
    phone = db.Column(db.String(20), default='')
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    students_taught = db.relationship('StudentTeacher', backref='teacher', lazy='dynamic')
    students_parented = db.relationship('StudentParent', backref='parent', lazy='dynamic')
    emotion_sessions = db.relationship('EmotionSession', backref='teacher', lazy='dynamic')
    
    def set_password(self, password: str) -> None:
        """Hash and set the user's password."""
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        """Verify a password against the stored hash."""
        if not self.password_hash:
            return False
        return check_password_hash(self.password_hash, password)

    def to_dict(self):
        """Convert ke dictionary untuk JSON response"""
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'full_name': self.full_name,
            'role': self.role,
            'phone': self.phone,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

class Student(db.Model):
    __tablename__ = 'students'
    
    id = db.Column(db.Integer, primary_key=True)
    student_code = db.Column(db.String(20), unique=True, nullable=False)
    full_name = db.Column(db.String(100), nullable=False)
    class_name = db.Column(db.String(50), nullable=False)
    birth_date = db.Column(db.Date)
    address = db.Column(db.Text)
    phone = db.Column(db.String(20))
    email = db.Column(db.String(120))
    subject = db.Column(db.String(100))
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    teachers = db.relationship('StudentTeacher', backref='student', lazy='dynamic')
    parents = db.relationship('StudentParent', backref='student', lazy='dynamic')
    emotion_sessions = db.relationship('EmotionSession', backref='student', lazy='dynamic')
    emotion_logs = db.relationship('EmotionLog', backref='student', lazy='dynamic')
    
    def to_dict(self):
        """Convert ke dictionary untuk JSON response"""
        return {
            'id': self.id,
            'student_code': self.student_code,
            'full_name': self.full_name,
            'class_name': self.class_name,
            'birth_date': self.birth_date.isoformat() if self.birth_date else None,
            'address': self.address,
            'phone': self.phone,
            'email': self.email,
            'subject': self.subject,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

class StudentTeacher(db.Model):
    __tablename__ = 'student_teachers'
    
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey('students.id'), nullable=False)
    teacher_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    __table_args__ = (db.UniqueConstraint('student_id', 'teacher_id', name='unique_student_teacher'),)

class StudentParent(db.Model):
    __tablename__ = 'student_parents'
    
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey('students.id'), nullable=False)
    parent_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    relationship = db.Column(db.String(50), default='parent')  # parent, guardian, etc.
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    __table_args__ = (db.UniqueConstraint('student_id', 'parent_id', name='unique_student_parent'),)

class EmotionSession(db.Model):
    __tablename__ = 'emotion_sessions'
    
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey('students.id'), nullable=True)
    teacher_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    session_name = db.Column(db.String(100))
    start_time = db.Column(db.DateTime, default=datetime.utcnow)
    end_time = db.Column(db.DateTime)
    status = db.Column(db.Enum('active', 'completed', 'cancelled'), default='active')
    notes = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    emotion_logs = db.relationship('EmotionLog', backref='session', lazy='dynamic')
    
    def to_dict(self):
        """Convert ke dictionary untuk JSON response"""
        return {
            'id': self.id,
            'student_id': self.student_id,
            'teacher_id': self.teacher_id,
            'session_name': self.session_name,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'status': self.status,
            'notes': self.notes,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class EmotionLog(db.Model):
    __tablename__ = 'emotion_logs'
    
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('emotion_sessions.id'), nullable=False)
    student_id = db.Column(db.Integer, db.ForeignKey('students.id'), nullable=False)
    emotion = db.Column(db.String(20), nullable=False)
    confidence_score = db.Column(db.Numeric(5, 4))
    image_path = db.Column(db.String(255))
    detected_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        """Convert ke dictionary untuk JSON response"""
        return {
            'id': self.id,
            'session_id': self.session_id,
            'student_id': self.student_id,
            'emotion': self.emotion,
            'confidence_score': float(self.confidence_score) if self.confidence_score else None,
            'image_path': self.image_path,
            'detected_at': self.detected_at.isoformat() if self.detected_at else None
        }

class EmotionAggregation(db.Model):
    """Tabel untuk menyimpan agregasi emosi harian per guru"""
    __tablename__ = 'emotion_aggregations'
    
    id = db.Column(db.Integer, primary_key=True)
    teacher_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    date = db.Column(db.Date, nullable=False)
    emotion = db.Column(db.String(20), nullable=False)
    count = db.Column(db.Integer, default=0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Unique constraint untuk mencegah duplikasi
    __table_args__ = (db.UniqueConstraint('teacher_id', 'date', 'emotion', name='unique_teacher_date_emotion'),)
    
    def to_dict(self):
        """Convert ke dictionary untuk JSON response"""
        return {
            'id': self.id,
            'teacher_id': self.teacher_id,
            'date': self.date.isoformat() if self.date else None,
            'emotion': self.emotion,
            'count': self.count,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

class Report(db.Model):
    __tablename__ = 'reports'
    
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    content = db.Column(db.Text)
    report_type = db.Column(db.String(50))  # daily, weekly, monthly, etc.
    generated_by = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    student_id = db.Column(db.Integer, db.ForeignKey('students.id'))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        """Convert ke dictionary untuk JSON response"""
        return {
            'id': self.id,
            'title': self.title,
            'content': self.content,
            'report_type': self.report_type,
            'generated_by': self.generated_by,
            'student_id': self.student_id,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
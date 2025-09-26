from flask import Blueprint, request, jsonify, current_app
from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity, get_jwt
from werkzeug.security import check_password_hash
from models import db, User
from datetime import datetime, timedelta
from functools import wraps
import re

auth_bp = Blueprint('auth', __name__)

def validate_email(email):
    """Validasi format email"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def validate_password(password):
    """Validasi password strength"""
    if len(password) < 8:
        return False, "Password minimal 8 karakter"
    if not re.search(r'[A-Z]', password):
        return False, "Password harus mengandung huruf besar"
    if not re.search(r'[a-z]', password):
        return False, "Password harus mengandung huruf kecil"
    if not re.search(r'\d', password):
        return False, "Password harus mengandung angka"
    return True, "Password valid"

@auth_bp.route('/register', methods=['POST'])
def register():
    """Endpoint untuk registrasi user baru"""
    try:
        data = request.get_json()
        
        # Validasi input
        required_fields = ['username', 'email', 'password', 'role', 'full_name']
        for field in required_fields:
            if field not in data or not data[field]:
                return jsonify({'error': f'Field {field} harus diisi'}), 400
        
        # Validasi email format
        if not validate_email(data['email']):
            return jsonify({'error': 'Format email tidak valid'}), 400
        
        # Validasi password strength
        is_valid_password, password_msg = validate_password(data['password'])
        if not is_valid_password:
            return jsonify({'error': password_msg}), 400
        
        # Validasi role
        if data['role'] not in ['guru', 'orang_tua', 'admin']:
            return jsonify({'error': 'Role harus guru, orang_tua, atau admin'}), 400
        
        # Cek apakah username sudah ada
        if User.query.filter_by(username=data['username']).first():
            return jsonify({'error': 'Username sudah digunakan'}), 409
        
        # Cek apakah email sudah ada
        if User.query.filter_by(email=data['email']).first():
            return jsonify({'error': 'Email sudah digunakan'}), 409
        
        # Buat user baru
        user = User(
            username=data['username'],
            email=data['email'],
            role=data['role'],
            full_name=data['full_name'],
            phone=data.get('phone', '')
        )
        user.set_password(data['password'])
        
        db.session.add(user)
        db.session.commit()
        
        return jsonify({
            'message': 'User berhasil didaftarkan',
            'user': user.to_dict()
        }), 201
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Terjadi kesalahan: {str(e)}'}), 500

@auth_bp.route('/login', methods=['POST'])
def login():
    """Endpoint untuk login user"""
    try:
        data = request.get_json()
        
        # Validasi input
        if not data or 'username' not in data or 'password' not in data:
            return jsonify({'error': 'Username dan password harus diisi'}), 400
        
        # Cari user berdasarkan username atau email
        user = User.query.filter(
            (User.username == data['username']) | (User.email == data['username'])
        ).first()
        
        if not user:
            return jsonify({'error': 'Username atau password salah'}), 401
        
        # Cek password
        if not user.check_password(data['password']):
            return jsonify({'error': 'Username atau password salah'}), 401
        
        # Cek apakah user aktif
        if not user.is_active:
            return jsonify({'error': 'Akun tidak aktif'}), 401
        
        # Generate JWT token (identity harus string di beberapa versi Flask-JWT-Extended)
        token = create_access_token(identity=str(user.id))
        
        return jsonify({
            'message': 'Login berhasil',
            'access_token': token,
            'user': user.to_dict()
        }), 200
        
    except Exception as e:
        return jsonify({'error': f'Terjadi kesalahan: {str(e)}'}), 500

@auth_bp.route('/profile', methods=['GET'])
@jwt_required()
def get_profile():
    """Endpoint untuk mendapatkan profil user yang sedang login"""
    try:
        user_id = get_jwt_identity()
        user = User.query.get(user_id)
        
        if not user:
            return jsonify({'error': 'User tidak ditemukan'}), 404
        
        return jsonify({
            'user': user.to_dict()
        }), 200
        
    except Exception as e:
        return jsonify({'error': f'Terjadi kesalahan: {str(e)}'}), 500

@auth_bp.route('/profile', methods=['PUT'])
@jwt_required()
def update_profile():
    """Endpoint untuk update profil user"""
    try:
        user_id = get_jwt_identity()
        user = User.query.get(user_id)
        
        if not user:
            return jsonify({'error': 'User tidak ditemukan'}), 404
        
        data = request.get_json()
        
        # Update field yang boleh diubah
        if 'full_name' in data:
            user.full_name = data['full_name']
        if 'phone' in data:
            user.phone = data['phone']
        if 'email' in data:
            if not validate_email(data['email']):
                return jsonify({'error': 'Format email tidak valid'}), 400
            # Cek apakah email sudah digunakan user lain
            existing_user = User.query.filter(
                User.email == data['email'],
                User.id != user_id
            ).first()
            if existing_user:
                return jsonify({'error': 'Email sudah digunakan'}), 409
            user.email = data['email']
        
        user.updated_at = datetime.utcnow()
        db.session.commit()
        
        return jsonify({
            'message': 'Profil berhasil diperbarui',
            'user': user.to_dict()
        }), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Terjadi kesalahan: {str(e)}'}), 500

@auth_bp.route('/change-password', methods=['POST'])
@jwt_required()
def change_password():
    """Endpoint untuk ganti password"""
    try:
        user_id = get_jwt_identity()
        user = User.query.get(user_id)
        
        if not user:
            return jsonify({'error': 'User tidak ditemukan'}), 404
        
        data = request.get_json()
        
        if 'old_password' not in data or 'new_password' not in data:
            return jsonify({'error': 'Old password dan new password harus diisi'}), 400
        
        # Cek old password
        if not user.check_password(data['old_password']):
            return jsonify({'error': 'Password lama salah'}), 401
        
        # Validasi new password
        is_valid_password, password_msg = validate_password(data['new_password'])
        if not is_valid_password:
            return jsonify({'error': password_msg}), 400
        
        # Set new password
        user.set_password(data['new_password'])
        user.updated_at = datetime.utcnow()
        db.session.commit()
        
        return jsonify({'message': 'Password berhasil diubah'}), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Terjadi kesalahan: {str(e)}'}), 500

@auth_bp.route('/logout', methods=['POST'])
@jwt_required()
def logout():
    """Endpoint untuk logout (client-side token removal)"""
    return jsonify({'message': 'Logout berhasil'}), 200

# Middleware untuk role-based access control
def require_role(required_roles):
    """Decorator untuk memerlukan role tertentu"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            user_id = get_jwt_identity()
            user = User.query.get(user_id)
            
            if not user:
                return jsonify({'error': 'User tidak ditemukan'}), 404
            
            if user.role not in required_roles:
                return jsonify({'error': 'Akses ditolak. Role tidak sesuai'}), 403
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator
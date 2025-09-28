"""
Helper functions untuk validasi input dan error handling
"""
from flask import jsonify
from typing import Dict, Any, Optional, List, Tuple
import re
from datetime import datetime

class ValidationError(Exception):
    """Custom exception untuk validation errors"""
    def __init__(self, message: str, field: str = None):
        self.message = message
        self.field = field
        super().__init__(self.message)

def validate_required_fields(data: Dict[str, Any], required_fields: List[str]) -> None:
    """
    Validasi field yang wajib diisi
    
    Args:
        data: Dictionary data yang akan divalidasi
        required_fields: List field yang wajib ada
        
    Raises:
        ValidationError: Jika ada field yang kosong
    """
    missing_fields = []
    
    for field in required_fields:
        if field not in data or data[field] is None or data[field] == "":
            missing_fields.append(field)
    
    if missing_fields:
        raise ValidationError(
            f"Field berikut wajib diisi: {', '.join(missing_fields)}",
            field="required_fields"
        )

def validate_email(email: str) -> bool:
    """
    Validasi format email
    
    Args:
        email: String email yang akan divalidasi
        
    Returns:
        bool: True jika valid, False jika tidak
    """
    if not email:
        return False
        
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def validate_password_strength(password: str) -> Tuple[bool, List[str]]:
    """
    Validasi kekuatan password
    
    Args:
        password: String password yang akan divalidasi
        
    Returns:
        Tuple[bool, List[str]]: (is_valid, list_of_errors)
    """
    errors = []
    
    if not password:
        errors.append("Password tidak boleh kosong")
        return False, errors
    
    if len(password) < 8:
        errors.append("Password minimal 8 karakter")
    
    if not re.search(r'[A-Z]', password):
        errors.append("Password harus mengandung minimal 1 huruf besar")
    
    if not re.search(r'[a-z]', password):
        errors.append("Password harus mengandung minimal 1 huruf kecil")
    
    if not re.search(r'\d', password):
        errors.append("Password harus mengandung minimal 1 angka")
    
    return len(errors) == 0, errors

def validate_phone_number(phone: str) -> bool:
    """
    Validasi format nomor telepon Indonesia
    
    Args:
        phone: String nomor telepon yang akan divalidasi
        
    Returns:
        bool: True jika valid, False jika tidak
    """
    if not phone:
        return False
    
    # Hapus semua karakter non-digit
    clean_phone = re.sub(r'\D', '', phone)
    
    # Cek apakah dimulai dengan 08 atau +62
    if clean_phone.startswith('08'):
        return len(clean_phone) >= 10 and len(clean_phone) <= 13
    elif clean_phone.startswith('62'):
        return len(clean_phone) >= 11 and len(clean_phone) <= 14
    else:
        return False

def validate_student_code(student_code: str) -> bool:
    """
    Validasi format student code
    
    Args:
        student_code: String student code yang akan divalidasi
        
    Returns:
        bool: True jika valid, False jika tidak
    """
    if not student_code:
        return False
    
    # Student code harus alphanumeric dan minimal 3 karakter
    return re.match(r'^[a-zA-Z0-9]{3,}$', student_code) is not None

def validate_role(role: str, allowed_roles: List[str]) -> bool:
    """
    Validasi role user
    
    Args:
        role: String role yang akan divalidasi
        allowed_roles: List role yang diizinkan
        
    Returns:
        bool: True jika valid, False jika tidak
    """
    return role in allowed_roles

def validate_relationship(relationship: str) -> bool:
    """
    Validasi jenis relasi parent-student
    
    Args:
        relationship: String relasi yang akan divalidasi
        
    Returns:
        bool: True jika valid, False jika tidak
    """
    allowed_relationships = ['ayah', 'ibu', 'wali', 'kakak', 'adik', 'kakek', 'nenek']
    return relationship.lower() in allowed_relationships

def validate_date_format(date_string: str, format: str = '%Y-%m-%d') -> bool:
    """
    Validasi format tanggal
    
    Args:
        date_string: String tanggal yang akan divalidasi
        format: Format tanggal yang diharapkan
        
    Returns:
        bool: True jika valid, False jika tidak
    """
    if not date_string:
        return False
    
    try:
        datetime.strptime(date_string, format)
        return True
    except ValueError:
        return False

def validate_positive_integer(value: Any, field_name: str = "value") -> int:
    """
    Validasi integer positif
    
    Args:
        value: Value yang akan divalidasi
        field_name: Nama field untuk error message
        
    Returns:
        int: Integer value yang sudah divalidasi
        
    Raises:
        ValidationError: Jika value tidak valid
    """
    try:
        int_value = int(value)
        if int_value <= 0:
            raise ValidationError(f"{field_name} harus berupa angka positif", field_name)
        return int_value
    except (ValueError, TypeError):
        raise ValidationError(f"{field_name} harus berupa angka yang valid", field_name)

def validate_boolean(value: Any, field_name: str = "value") -> bool:
    """
    Validasi boolean value
    
    Args:
        value: Value yang akan divalidasi
        field_name: Nama field untuk error message
        
    Returns:
        bool: Boolean value yang sudah divalidasi
        
    Raises:
        ValidationError: Jika value tidak valid
    """
    if isinstance(value, bool):
        return value
    elif isinstance(value, str):
        if value.lower() in ['true', '1', 'yes', 'on']:
            return True
        elif value.lower() in ['false', '0', 'no', 'off']:
            return False
        else:
            raise ValidationError(f"{field_name} harus berupa true/false", field_name)
    elif isinstance(value, int):
        return bool(value)
    else:
        raise ValidationError(f"{field_name} harus berupa boolean", field_name)

def create_error_response(error_message: str, status_code: int = 400, field: str = None) -> Tuple[Dict[str, Any], int]:
    """
    Buat response error yang konsisten
    
    Args:
        error_message: Pesan error
        status_code: HTTP status code
        field: Nama field yang error (optional)
        
    Returns:
        Tuple[Dict[str, Any], int]: (response_dict, status_code)
    """
    response = {
        'error': error_message,
        'success': False
    }
    
    if field:
        response['field'] = field
    
    return response, status_code

def handle_validation_error(error: ValidationError) -> Tuple[Dict[str, Any], int]:
    """
    Handle ValidationError dan buat response yang sesuai
    
    Args:
        error: ValidationError instance
        
    Returns:
        Tuple[Dict[str, Any], int]: (response_dict, status_code)
    """
    return create_error_response(error.message, 400, error.field)

def validate_json_data(data: Any, required_fields: List[str] = None) -> Dict[str, Any]:
    """
    Validasi data JSON dan field yang diperlukan
    
    Args:
        data: Data yang akan divalidasi
        required_fields: List field yang wajib ada
        
    Returns:
        Dict[str, Any]: Data yang sudah divalidasi
        
    Raises:
        ValidationError: Jika data tidak valid
    """
    if not isinstance(data, dict):
        raise ValidationError("Data harus berupa JSON object")
    
    if required_fields:
        validate_required_fields(data, required_fields)
    
    return data

def sanitize_string(value: str, max_length: int = None) -> str:
    """
    Sanitasi string input
    
    Args:
        value: String yang akan disanitasi
        max_length: Panjang maksimal string
        
    Returns:
        str: String yang sudah disanitasi
    """
    if not value:
        return ""
    
    # Hapus whitespace di awal dan akhir
    sanitized = value.strip()
    
    # Batasi panjang jika diperlukan
    if max_length and len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
    
    return sanitized

def validate_file_upload(filename: str, allowed_extensions: List[str], max_size: int = None) -> bool:
    """
    Validasi file upload
    
    Args:
        filename: Nama file yang akan divalidasi
        allowed_extensions: List ekstensi yang diizinkan
        max_size: Ukuran maksimal file dalam bytes
        
    Returns:
        bool: True jika valid, False jika tidak
    """
    if not filename:
        return False
    
    # Cek ekstensi file
    if '.' not in filename:
        return False
    
    file_extension = filename.rsplit('.', 1)[1].lower()
    if file_extension not in allowed_extensions:
        return False
    
    return True
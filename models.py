from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
import datetime

db = SQLAlchemy()

class User(UserMixin, db.Model):
    """Mô hình User để lưu thông tin người dùng"""
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    phone = db.Column(db.String(15), nullable=True)
    email_verified = db.Column(db.Boolean, default=False)
    verification_code = db.Column(db.String(6), nullable=True)
    code_expiry = db.Column(db.DateTime, nullable=True)
    questions_asked = db.Column(db.Integer, default=0)
    subscription_expiry = db.Column(db.DateTime, nullable=True)
    subscription_type = db.Column(db.String(20), nullable=True)
    
    def set_password(self, password):
        """Mã hóa và lưu mật khẩu"""
        self.password_hash = generate_password_hash(password)
        
    def check_password(self, password):
        """Kiểm tra mật khẩu"""
        return check_password_hash(self.password_hash, password)
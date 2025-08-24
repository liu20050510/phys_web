from flask_sqlalchemy import SQLAlchemy
import uuid

db = SQLAlchemy()

class HealthHistory(db.Model):
    __tablename__ = 'health_history'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(36), db.ForeignKey('users.user_id'), nullable=False)
    username = db.Column(db.String(80), nullable=False)
    detect_time = db.Column(db.DateTime)
    bmp = db.Column(db.Double())
    image_path = db.Column(db.String(255))
    csv_path = db.Column(db.String(255))
    user = db.relationship('User', back_populates="health_records")

class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(36), unique=True, nullable=False, default=lambda: str(uuid.uuid4()))
    username = db.Column(db.String(80), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    health_records = db.relationship(
        'HealthHistory',
        back_populates="user",
        lazy=True
    )
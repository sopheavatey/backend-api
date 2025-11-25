from datetime import timedelta, datetime
from passlib.context import CryptContext
from jose import jwt
from core.config import settings
from sqlalchemy.orm import Session
from middleware.auth.models import Users

# Use settings for configuration
SECRET_KEY = settings.JWT_SECRET_KEY
ALGORITHM = settings.ALGORITHM
ACCESS_TOKEN_EXPIRE_MINUTES = settings.ACCESS_TOKEN_EXPIRE_MINUTES

bcrypt_context = CryptContext(schemes=['bcrypt'], deprecated='auto')

def hash_password(password: str) -> str:
    """Hashes the plain text password."""
    return bcrypt_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies a plain text password against the hashed password."""
    return bcrypt_context.verify(plain_password, hashed_password)

def authenticate_user(username_or_email: str, password: str, db: Session) -> Users | bool:
    """Authenticates a user by username/email and password."""
    user = db.query(Users).filter(
        (Users.username == username_or_email) | (Users.email == username_or_email)
    ).first()

    if not user:
        return False
    
    if not verify_password(password, user.password_hash):
        return False
    
    return user

def create_access_token(username: str, user_id: int, email: str, expires_delta: timedelta | None = None) -> str:
    """Creates a JWT access token."""
    encode = {'sub': username, 'id': user_id, 'email': email}
    
    if expires_delta:
        expires = datetime.utcnow() + expires_delta
    else:
        expires = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        
    encode.update({'exp': expires})
    
    return jwt.encode(encode, SECRET_KEY, algorithm=ALGORITHM)
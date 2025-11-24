from datetime import timedelta, datetime
from typing import Annotated
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session
from database import SessionLocal
from models import Users
from passlib.context import CryptContext
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from jose import jwt, JWTError
from dotenv import load_dotenv
import os
load_dotenv()

router = APIRouter(
    prefix= '/auth',
    tags= ['auth'],
)

SECRET_KEY = os.getenv("JWT_SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM")

bcrypt_context = CryptContext(schemes=['bcrypt'], deprecated='auto')
oauth2_bearer = OAuth2PasswordBearer(tokenUrl='/auth/token')

class CreateUserRequest(BaseModel):
    username: str
    email: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

db_dependency = Annotated[Session, Depends(get_db)]

@router.post("/register", status_code=status.HTTP_201_CREATED) 
async def create_user(db:db_dependency,
                      create_user_request: CreateUserRequest):
    # Check if username or email already exists
    if db.query(Users).filter((Users.username == create_user_request.username) | (Users.email == create_user_request.email)).first():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username or email already registered.")
    
    create_user_model = Users(
        username = create_user_request.username,
        email = create_user_request.email,
        hashed_password = bcrypt_context.hash(create_user_request.password),
    )

    db.add(create_user_model)
    db.commit()
    return {"message": "User successfully registered"}

@router.post("/token", response_model=Token)
async def login_for_access_token(form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
                                 db: db_dependency):
    
    user = authenticate_user(form_data.username, form_data.password, db)

    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail = 'Incorrect username or email or password.')
    
    token = create_access_token(user.username, user.id, user.email, timedelta(minutes=30)) 

    return {'access_token': token, 'token_type': 'bearer'}


def authenticate_user(username_or_email: str, password: str, db: Session):
    user = db.query(Users).filter(
        (Users.username == username_or_email) | (Users.email == username_or_email)
    ).first()

    if not user:
        return False
    
    if not bcrypt_context.verify(password, user.hashed_password):
        return False
    
    return user

def create_access_token(username: str, user_id: int, email: str, expires_delta: timedelta):
    encode = {'sub': username, 'id':user_id, 'email': email}
    expires = datetime.utcnow() + expires_delta
    encode.update({'exp': expires})
    return jwt.encode(encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: Annotated [str, Depends(oauth2_bearer)]) :
    try:
        payload = jwt. decode (token, SECRET_KEY, algorithms= [ALGORITHM] )
        username: str = payload.get ('sub' )
        user_id: int = payload.get ('id')
        email: str = payload.get('email')

        if username is None or user_id is None or email is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                                detail='Could not validate user.')
        return {'username': username, 'id': user_id, 'email': email}
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail= 'Could not validate user.')
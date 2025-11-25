from datetime import timedelta
from typing import Annotated
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from schemas.auth_schema import CreateUserRequest, Token
from middleware.auth.database import SessionLocal
from middleware.auth.models import Users
from middleware.auth.auth_service import authenticate_user, create_access_token, hash_password
from middleware.auth.auth_deps import db_dependency, user_dependency

router = APIRouter(
    prefix= '/auth',
    tags= ['auth'],
)

@router.post("/register", status_code=status.HTTP_201_CREATED) 
async def create_user(db: db_dependency,
                      create_user_request: CreateUserRequest):
    """Register a new user."""
    # Check if username or email already exists
    if db.query(Users).filter(
        (Users.username == create_user_request.username) | 
        (Users.email == create_user_request.email)
    ).first():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="Username or email already registered."
        )
    
    # Create the new user model
    create_user_model = Users(
        username = create_user_request.username,
        email = create_user_request.email,
        hashed_password = hash_password(create_user_request.password), # Use the service function
    )

    db.add(create_user_model)
    db.commit()
    return {"message": "User successfully registered"}

@router.post("/token", response_model=Token)
async def login_for_access_token(form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
                                 db: db_dependency):
    """Login and get an access token."""
    
    # Use the service function for authentication
    user = authenticate_user(form_data.username, form_data.password, db)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail = 'Incorrect username or email or password.',
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create token, timedelta is passed as None to use the default from settings
    token = create_access_token(user.username, user.id, user.email, expires_delta=None) 

    return {'access_token': token, 'token_type': 'bearer'}

# Simple test endpoint to verify authentication is working
@router.get("/me", status_code=status.HTTP_200_OK)
async def get_user_info(user: user_dependency):
    """Returns the current authenticated user's data."""
    return {"user": user}
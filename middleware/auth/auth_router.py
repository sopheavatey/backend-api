from datetime import timedelta
from typing import Annotated
from fastapi import APIRouter, Depends, HTTPException, status, Response
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from schemas.auth_schema import CreateUserRequest, Token
from middleware.auth.database import SessionLocal
from middleware.auth.models import Users
from middleware.auth.auth_service import authenticate_user, create_access_token, hash_password
from middleware.auth.auth_deps import db_dependency, user_dependency
from core.config import settings

router = APIRouter(
    prefix= '/auth',
    tags= ['auth'],
)

@router.post("/register", status_code=status.HTTP_201_CREATED) 
async def create_user(db: db_dependency,
                      create_user_request: CreateUserRequest):
    if db.query(Users).filter(
        (Users.username == create_user_request.username) | 
        (Users.email == create_user_request.email)
    ).first():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="Username or email already registered."
        )
    
    create_user_model = Users(
        username = create_user_request.username,
        email = create_user_request.email,
        password_hash = hash_password(create_user_request.password),
    )

    db.add(create_user_model)
    db.commit()
    return {"message": "User successfully registered"}

@router.post("/token", response_model=Token)
async def login_for_access_token(
    response: Response,
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
    db: db_dependency
):
    user = authenticate_user(form_data.username, form_data.password, db)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail = 'Incorrect username or email or password.',
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    token_str = create_access_token(user.username, user.id, user.email, expires_delta=None) 

    # We set BOTH cookie and return token body to support both auth methods
    response.set_cookie(
        key="access_token",
        value=token_str,
        httponly=True,
        secure=False, # Set True in Prod
        samesite="lax",
        max_age=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )

    return {'access_token': token_str, 'token_type': 'bearer'}

@router.post("/logout")
async def logout(response: Response):
    response.delete_cookie(key="access_token")
    return {"message": "Successfully logged out"}

@router.get("/me", status_code=status.HTTP_200_OK)
async def get_user_info(user: user_dependency):
    # Return user data directly so frontend can use it
    return {"user": user}
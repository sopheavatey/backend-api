from typing import Annotated, Generator
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError
from sqlalchemy.orm import Session

from core.config import settings
from middleware.auth.database import get_db
from schemas.auth_schema import UserData

# Dependencies
db_dependency = Annotated[Session, Depends(get_db)]
oauth2_bearer = OAuth2PasswordBearer(tokenUrl='/auth/token', auto_error=False)


def get_token_from_cookie_or_header(request: Request, token_header: str | None = Depends(oauth2_bearer)) -> str:
    """
    Custom dependency to retrieve the token.
    Priority 1: HttpOnly Cookie (access_token)
    Priority 2: Authorization Header (Bearer ...)
    """
    # 1. Try to get from Cookie
    token_cookie = request.cookies.get("access_token")
    if token_cookie:
        return token_cookie
    
    # 2. Try to get from Header (if cookie missing)
    if token_header:
        return token_header
        
    # 3. If neither, raise error
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Not authenticated. No token found in cookie or header.",
    )


async def get_current_user(token: Annotated[str, Depends(oauth2_bearer)]) -> UserData:
    """
    Decodes the JWT token and returns user data.
    Raises 401 if validation fails.
    """
    try:
        # Decode the token using the secret key and algorithm from settings
        payload = jwt.decode(
            token, 
            settings.JWT_SECRET_KEY, 
            algorithms=[settings.ALGORITHM]
        )
        
        # Extract fields
        username: str = payload.get('sub')
        user_id: int = payload.get('id')
        email: str = payload.get('email')

        if username is None or user_id is None or email is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail='Could not validate credentials: Missing token payload data.'
            )
            
        # Return the validated user data as a Pydantic model
        return UserData(username=username, id=user_id, email=email)
        
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='Could not validate credentials: Invalid token signature or format.'
        )

# Dependency for routes that require authentication
user_dependency = Annotated[UserData, Depends(get_current_user)]
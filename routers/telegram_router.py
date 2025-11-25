from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel

from middleware.auth.auth_deps import user_dependency, db_dependency
from middleware.auth.models import Users

router = APIRouter(
    prefix="/telegram",
    tags=["telegram"],
)

class LinkTelegramRequest(BaseModel):
    # This token comes from the URL the bot sent: ?token=...
    telegram_token: str 

@router.post("/link")
async def link_telegram_account(
    request: LinkTelegramRequest, 
    user_data: user_dependency, # Ensures user is logged in via website
    db: db_dependency
):
    """
    Links the currently logged-in web user to a Telegram Chat ID.
    The telegram_token contains the encrypted/signed chat_id.
    """
    try:
        # In a real app, verify signature. 
        # For simplicity, we assume the token IS the chat_id (see security note below).
        chat_id = int(request.telegram_token) 
        
        # 1. Check if this chat_id is already linked to another user
        existing = db.query(Users).filter(Users.telegram_chat_id == chat_id).first()
        if existing:
            raise HTTPException(status_code=400, detail="This Telegram account is already linked.")

        # 2. Update the current user
        user_db = db.query(Users).filter(Users.id == user_data.id).first()
        user_db.telegram_chat_id = chat_id
        db.commit()
        
        return {"message": "Telegram account successfully connected!"}

    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid token format")
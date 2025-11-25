from sqlalchemy import Column, Integer, String, BigInteger
from middleware.auth.database import Base 

class Users(Base):
    __tablename__= 'users'

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True)
    email = Column(String, unique=True)
    password_hash = Column(String)
    # New column to link Telegram accounts
    telegram_chat_id = Column(BigInteger, unique=True, nullable=True)
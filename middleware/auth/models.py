from sqlalchemy import Column, Integer, String 
from middleware.auth.database import Base 

class Users(Base):
    __tablename__= 'users'

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True)
    email = Column(String, unique=True)
    hashed_password = Column(String)
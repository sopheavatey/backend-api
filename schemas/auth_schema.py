from pydantic import BaseModel

class CreateUserRequest(BaseModel):
    username: str
    email: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

# Schema for the user object returned by the dependency
class UserData(BaseModel):
    username: str
    id: int
    email: str
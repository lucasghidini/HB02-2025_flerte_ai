from pydantic import BaseModel
from typing import List, Optional

class MessageHistory(BaseModel):
    role: str
    content: str
    image_data: Optional[bytes] = None

class UserPreferences(BaseModel):
    style: str
    length: str

class ConversationRequest(BaseModel):
    history: List[MessageHistory]
    preferences: UserPreferences
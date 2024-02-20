from pydantic import BaseModel


class Role(str):
    user = "user"
    system = "system"
    assistant = "assistant"

from dataclasses import dataclass
from typing import List, Optional, Union


@dataclass
class Message:
    role: str = ""
    content: str = ""


@dataclass
class Choice:
    index: int
    message: Message


@dataclass
class StreamChoice:
    index: int
    delta: Message


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class BaseResponse:
    code: int = 0
    error_message: str = ""
    id: Optional[Union[str, float]] = ""
    created: str = None


@dataclass
class ChatResponse(BaseResponse):
    object: str = "chat.completion"
    choices: List[Union[Choice, StreamChoice]] = None
    usage: Optional[Usage] = None

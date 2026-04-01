from pydantic import BaseModel
from typing import Dict, Any


class FunctionParameter(BaseModel):
    type: str


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: Dict[str, FunctionParameter]
    returns: Dict[str, str]


class FunctionTest(BaseModel):
    prompt: str


class FunctionCall(BaseModel):
    function_name: str
    arguments: Dict[str, Any]

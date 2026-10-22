from pydantic import BaseModel
from typing import Dict, Any


class FunctionParameter(BaseModel):
    """Represents a parameter for a function definition.

    Attributes:
        type (str): The data type of the parameter.
    """
    type: str


class FunctionDefinition(BaseModel):
    """Defines a function with its name, description, parameters, and return
    types.

    Attributes:
        name (str): The name of the function.
        description (str): A description of what the function does.
        parameters (Dict[str, FunctionParameter]): The parameters of the
        function.
        returns (Dict[str, str]): The return types of the function.
    """
    name: str
    description: str
    parameters: Dict[str, FunctionParameter]
    returns: Dict[str, str]


class FunctionTest(BaseModel):
    """Represents a test case for function calling.

    Attributes:
        prompt (str): The prompt used for the test.
    """
    prompt: str


class FunctionCall(BaseModel):
    """Represents a function call with its name and arguments.

    Attributes:
        function_name (str): The name of the function to call.
        arguments (Dict[str, Any]): The arguments for the function call.
    """
    function_name: str
    arguments: Dict[str, Any]

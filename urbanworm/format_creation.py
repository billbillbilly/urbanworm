from pydantic import BaseModel, create_model
from typing import List, TypeVar, Generic

# Generic type variable
T = TypeVar("T")

def schema(fields: dict):
    """
    Create a customized QnA model with customized fields.
    Parameters:
    - fields (dict): customized fields to add dynamically.
    example:
    fields = {
        "question": (str, ...),
        "answer": (str, ...),
        "explanation": (str, ...),
    }
    schema(fields)

    Returns:
    - Pydantic model: Customized QnA class.
    """

    CustomQnA = create_model("QnA", **fields)
    return CustomQnA

class Response(BaseModel, Generic[T]):
    responses: List[T]

def create_format(fields: dict):
    """
    format a Response model with a customized QnA model.
    
    Parameters:
    - fields (dict): fields to add dynamically.

    Returns:
    - Pydantic model: Response class.
    """

    # Dynamically create the model
    CustomQnA = schema(fields)

    return Response[CustomQnA]


from pydantic import BaseModel, create_model
from typing import List, TypeVar, Generic
from typing import Type

# Generic type variable
T = TypeVar("T")

def schema(fields: dict):
    """
    Create a customized QnA model with customized fields.

    Args:
        fields (dict): customized fields to add dynamically.
            example:
                fields = {
                    "question": (str, ...),
                    "answer": (str, ...),
                    "explanation": (str, ...),
                }
                schema(fields)

    Returns:
        Pydantic model: Customized QnA class.
    """

    CustomQnA = create_model("QnA", **fields)
    return CustomQnA

class Response(BaseModel, Generic[T]):
    responses: List[T]

def create_format(fields: dict) -> Type[Response]:
    """
    Create a generic `Response` model using a dynamically defined Pydantic schema.

    This function allows you to define a custom set of fields (e.g., for QnA-style data),
    and returns a typed `Response[CustomQnA]` model class.

    Args:
        fields (dict): A dictionary of field definitions for the inner model.
            Example:
                fields = {
                    "question": (str, ...),
                    "answer": (str, ...),
                    "explanation": (str, ...),
                }

    Returns:
        Type[Response]: A `Response` model class parameterized with the dynamically created schema.
    """

    # Dynamically create the model
    CustomQnA = schema(fields)
    return Response[CustomQnA]


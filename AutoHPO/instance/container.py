from typing import Annotated, List
from typing_extensions import TypedDict

class PathContaioner:




class DataContainer(TypedDict):
    sourceCode: Annotated[str, "sourceCode"]
    modelInfo : Annotated[str, "modelInfo"]
    hyperParams: Annotated[str, "hyperParams"]
    userPrompt: Annotated[str, "userPrompt"]
    refactoredCode: Annotated[str, "refactoredCode"]

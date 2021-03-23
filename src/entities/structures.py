from typing import Any
from typing import List
from typing import Tuple
from typing import NewType

"""
Making custom re-usable types for function declarations.
"""
Point = NewType('Point', Tuple[Any, Any])
ListPoint = NewType('ListPoint', List[Point])
ListPointClassified = NewType('ListPointClassified', List[Tuple[Point, int]])

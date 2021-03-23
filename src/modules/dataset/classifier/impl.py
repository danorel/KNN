import math

from src.entities.structures import Point
from src.modules.dataset.classifier.interface import AbstractPointClassifier


class PointClassifier(AbstractPointClassifier):
    def __init__(self, classes: int, probabilities: float = 1.):
        self.__classes = classes
        self.__probabilities = probabilities

    def commit(self, point: Point):
        return round(self.__decider(point))

    def __decider(self, point: Point) -> int:
        """
        Circle classifier.
        Decides to which part belongs current point.
        :return:
        """
        shift_degrees = 360. / self.__classes

        # Calculation of the point degrees by it's tangent.
        point_degrees = math.degrees(math.atan(math.fabs(point[1] / point[0])))

        # Case, when point belongs to the plane except first.
        if -1 < point[0] < 0 and  0 < point[1] < 1: point_degrees = 180 - point_degrees
        if -1 < point[0] < 0 and -1 < point[1] < 0: point_degrees += 180
        if  0 < point[0] < 1 and -1 < point[1] < 0: point_degrees = 360 - point_degrees

        # Variables meant for iteration inside the circle
        prev_degrees = 0
        next_degrees = shift_degrees

        for _, index in enumerate(range(1, self.__classes + 1)):
            if prev_degrees < point_degrees < next_degrees:
                return index

            prev_degrees = next_degrees
            next_degrees += shift_degrees

        return -1

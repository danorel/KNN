import math

from src.entities.structures import Point
from src.modules.dataset.classifier.interface import AbstractPointClassifier


class PointClassifier(AbstractPointClassifier):
    def __init__(self, classes: int, probabilities: float = 1.):
        self.__classes = classes
        self.__probabilities = probabilities

    def commit(self, point: Point):
        point_class = self.__decider(point)

        if point_class < 1: return 1
        if point_class > self.__classes: return self.__classes

        return round(point_class)

    def __decider(self, point: Point) -> int:
        """
        Circle classifier.
        Decides to which part belongs current point.
        :return:
        """
        degree = 360. / self.__classes
        length = round(degree)

        degree_prev = 0
        degree_next = degree

        for _, index in enumerate(range(1, length)):
            if ((0 < point[0] < math.cos(math.radians(degree_prev))) and
               (0 < point[1] < math.sin(math.radians(degree_next)))):
                return index

            degree_prev = degree_next
            degree_next += degree

        return -1

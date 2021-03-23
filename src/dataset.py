from math import floor
from numpy import random
from typing import Any
from typing import List
from typing import Tuple
from typing import NewType

"""
Making custom types for re-using in function declarations.
"""
Point = NewType('Point', Tuple[Any, Any])
ListPoint = NewType('ListPoint', List[Point])
ListPointClassified = NewType('ListPointClassified', List[Tuple[Point, int]])


class RandomPointDatasetGenerator:
    """
    Generate random floating point values
    """
    def __init__(self, size: int):
        self.__size = size

    """
    Method for generating the points without classes
    """
    def commit(self) -> ListPoint:
        training_set: ListPoint = []

        # Generate random numbers between 0-1
        for point in zip(random.rand(self.__size), random.rand(self.__size)):
            training_set.append(point)

        return training_set

    """
    Method for generating the classified random points
    """
    def commit_classified(self) -> ListPointClassified:
        training_set: ListPointClassified = []

        # Generate random numbers between 0-1
        for point in zip(self.__rand_negative_list(self.__size), self.__rand_negative_list(self.__size)):  # type: Point
            training_set.append((point, self.__classifier(point)))

        return training_set

    """
    Method for deciding to which class the point belongs.
    Using probability.
    """
    @staticmethod
    def __classifier(point: Point, probabilities=(.85, .05, .05, .05)):
        point_class = abs((point[0] + point[1]) / 2) + 1

        if point[0] > 0 and point[1] > 0:
            point_class *= probabilities[0] * 1 + probabilities[1] * 2 + probabilities[2] * 3 + probabilities[3] * 4

        if point[0] < 0 and point[1] > 0:
            point_class *= probabilities[0] * 1 + probabilities[1] * 1 + probabilities[2] * 3 + probabilities[3] * 4

        if point[0] < 0 and point[1] < 0:
            point_class *= probabilities[0] * 3 + probabilities[1] * 1 + probabilities[2] * 2 + probabilities[3] * 4

        if point[0] > 0 and point[1] < 0:
            point_class *= probabilities[0] * 4 + probabilities[1] * 1 + probabilities[2] * 2 + probabilities[3] * 3

        return floor(point_class)

    """
    Method for extracting the negative random list of values.
    """
    def __rand_negative_list(self, size: int):
        return [self.__rand_negative() for _ in range(size)]

    """
    Method for extracting the negative random value.
    Using numpy rand() for the purpose.
    """
    @staticmethod
    def __rand_negative(low=-1, high=1):
        return (high - low) * random.rand() + low


class RandomPointDatasetSplitter:
    def __init__(self, dataset: ListPoint or ListPointClassified):
        self.__dataset = dataset

    def commit(self, ratio=.7):
        x_train, x_test = [], []
        y_train, y_test = [], []

        for point in self.__dataset:
            if random.rand() < ratio:
                x_train.append(point[0])
                y_train.append(point[1])
            else:
                x_test.append(point[0])
                y_test.append(point[1])

        return x_train, x_test, y_train, y_test

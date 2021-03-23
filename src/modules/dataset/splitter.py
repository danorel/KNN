from numpy import random

from src.entities.structures import ListPoint
from src.entities.structures import ListPointClassified


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

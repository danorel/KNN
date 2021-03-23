from numpy import random

from src.entities.structures import Point
from src.entities.structures import ListPoint
from src.entities.structures import ListPointClassified

from src.modules.dataset.classifier.interface import AbstractPointClassifier
from src.modules.dataset.generator.interface import AbstractRandomValueGenerator
from src.modules.dataset.generator.interface import AbstractRandomDatasetGenerator
from src.modules.dataset.generator.interface import AbstractRandomListGenerator


class RandomValueGenerator(AbstractRandomValueGenerator):
    def __init__(self, minimum=-1, maximum=1):
        self.__minimum = minimum
        self.__maximum = maximum

    def rand(self):
        return random.uniform(self.__minimum, self.__maximum)


class RandomListGenerator(AbstractRandomListGenerator):
    def __init__(self, generator: AbstractRandomValueGenerator):
        self.__generator: AbstractRandomValueGenerator = generator

    def rand(self, size: int):
        return [self.__generator.rand() for _ in range(size)]


class RandomPointDatasetGenerator(AbstractRandomDatasetGenerator):
    def __init__(self, generator: AbstractRandomListGenerator, classifier: AbstractPointClassifier = None):
        self.__generator: AbstractRandomListGenerator = generator
        self.__classifier: AbstractPointClassifier = classifier

    def commit(self, size: int) -> ListPoint:
        training_set: ListPoint = []

        for point in zip(self.__generator.rand(size), self.__generator.rand(size)): # type: Point
            training_set.append(point)

        return training_set

    def commit_classified(self, size: int) -> ListPointClassified:
        training_set: ListPointClassified = []

        for point in zip(self.__generator.rand(size), self.__generator.rand(size)):  # type: Point
            training_set.append((point, self.__classifier.commit(point)))

        return training_set

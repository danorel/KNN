import abc

from src.entities.structures import Point


class AbstractPointClassifier(abc.ABC):
    @abc.abstractmethod
    def commit(self, point: Point):
        """
        Method for deciding to which type of class belongs current point.
        """
        pass

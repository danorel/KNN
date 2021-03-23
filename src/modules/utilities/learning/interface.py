import abc


class AbstractLearningKNN(abc.ABC):
    @abc.abstractmethod
    def learn_model(self) -> int:
        """
        Return the most efficient K-neighbour number.
        :return: int
        """
        pass

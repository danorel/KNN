import abc


class AbstractRandomDatasetSplitter(abc.ABC):
    @abc.abstractmethod
    def commit(self, ratio=.7):
        pass

import abc


class AbstractEstimatorKNN(abc.ABC):
    @abc.abstractmethod
    def leave_one_out(self, x, y) -> float:
        """
        Percentage of correct-giving results of the model.
        Using leave one out metrics.
        :return: float
        """
        pass

    def confusion_matrix(self, x, y) -> float:
        """
        Percentage of correct-giving results of the model.
        Using confusion matrix metrics.
        :return:
        """
        pass

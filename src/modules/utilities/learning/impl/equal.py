from src.modules.kit.interface import AbstractKNN
from src.modules.utilities.model.impl.equal import ModelEqualKNN
from src.modules.utilities.learning.interface import AbstractLearningKNN


class LearningKNNEqual(AbstractLearningKNN):
    def __init__(self, classifier: AbstractKNN, size: int, classes: int, max_k=20, ratio=.8):
        """
        Launches the KNN Model.
        :return: None
        """
        self.__size: int = size
        self.__ratio: float = ratio
        self.__max_k: int = max_k
        self.__classes: int = classes
        self.__dataset: list or None = None
        self.__classifier: AbstractKNN = classifier

    def learn_model(self) -> int:
        max_k = 0
        max_precision = 0

        for k in range(1, self.__max_k + 1):
            pipeline: ModelEqualKNN = ModelEqualKNN(self.__classifier, k, self.__size, self.__classes)
            x_train, x_test, y_train, y_test = pipeline.prepare_dataset()
            precision = pipeline.estimate(x_train, y_train)
            print("Precision on neighbours", k, "equals:", precision)
            if precision > max_precision:
                max_precision = precision
                max_k = k

        return max_k

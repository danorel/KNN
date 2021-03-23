from src.modules.kit.equal import KNNSciKit
from src.modules.kit.equal import KNNCustom
from src.modules.pipelines.configuration import LaunchKNNPipeline


class LearningKNNPipeline:
    def __init__(self, Model: KNNCustom or KNNSciKit, size: int, max_k=20, ratio=.8):
        """
        Launches the KNN Model.
        :return: None
        """
        self.__size: int = size
        self.__ratio: float = ratio
        self.__max_k: int = max_k
        self.__dataset: list or None = None
        self.__Model: KNNCustom or KNNSciKit = Model

    def learn_model(self) -> int:
        max_k = 0
        max_precision = 0

        for k in range(1, self.__max_k):
            pipeline: LaunchKNNPipeline = LaunchKNNPipeline(self.__Model, k, self.__size)
            x_train, x_test, y_train, y_test = pipeline.prepare_dataset()
            precision = pipeline.estimate(x_train, y_train)
            print("Precision on neighbours", k, "equals:", precision)
            if precision > max_precision:
                max_precision = precision
                max_k = k

        return max_k
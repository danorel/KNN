from src.modules.metrics.impl import EstimatorKNN
from src.modules.kit.interface import AbstractKNN
from src.modules.dataset.classifier.impl import PointClassifier
from src.modules.dataset.generator.impl import RandomValueGenerator
from src.modules.dataset.generator.impl import RandomListGenerator
from src.modules.dataset.generator.impl import RandomPointDatasetGenerator
from src.modules.dataset.splitter.impl import RandomPointDatasetSplitter
from src.modules.utilities.display.impl import DisplayKNN
from src.modules.utilities.model.interface import AbstractModelKNN


class ModelEqualKNN(AbstractModelKNN):
    def __init__(self, model: AbstractKNN, k: int, size: int, classes: int, ratio=.8):
        """
        Launches the KNN Model.
        :return: None
        """
        self.__k: int = k
        self.__size: int = size
        self.__ratio: float = ratio
        self.__classes: int = classes
        self.__dataset: list or None = None
        self.__model: AbstractKNN = model(k)

    def prepare_dataset(self):
        self.__dataset = RandomPointDatasetGenerator(
            classifier=PointClassifier(classes=self.__classes),
            generator=RandomListGenerator(
                generator=RandomValueGenerator(),)
        ).commit_classified(self.__size)
        return RandomPointDatasetSplitter(self.__dataset).commit(ratio=self.__ratio)

    def prepare_model(self, x_train, x_test, y_train):
        self.__model.fit_scaler(x_train, x_test)
        x_test  = self.__model.transform(x_test)
        x_train = self.__model.fit_transform(x_train)
        self.__model.fit_model(x_train, y_train)
        return x_train, x_test

    def commit_prediction(self, x_test):
        return self.__model.predict(x_test)

    def build_map(self, x, y, title) -> None:
        DisplayKNN(self.__model, x, y, self.__classes, title).render_map()
        return None

    def build_graph(self, x, y, title) -> None:
        DisplayKNN(self.__model, x, y, self.__classes, title).render_graph()
        return None

    def estimate(self, x, y) -> float:
        return EstimatorKNN(self.__model).confusion_matrix(x, y)

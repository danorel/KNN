from src.modules.utilities.display import KNNDashboard
from src.modules.metrics.impl import KNNEstimator
from src.modules.kit.interface import AbstractKNN
from src.modules.dataset.classifier.impl import PointClassifier
from src.modules.dataset.generator.impl import RandomValueGenerator
from src.modules.dataset.generator.impl import RandomListGenerator
from src.modules.dataset.generator.impl import RandomPointDatasetGenerator
from src.modules.dataset.splitter.impl import RandomPointDatasetSplitter


class KNNEqualPipeline:
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
        """
        Stage 1: Generate dataset with classes.
                 - Generate N points for the whole dataset.
                 - Divide the dataset by train/test.
        """
        self.__dataset = RandomPointDatasetGenerator(
            classifier=PointClassifier(classes=self.__classes),
            generator=RandomListGenerator(
                generator=RandomValueGenerator(),)
        ).commit_classified(self.__size)
        return RandomPointDatasetSplitter(self.__dataset).commit(ratio=self.__ratio)

    def prepare_model(self, x_train, x_test, y_train):
        """
        Stage 2: Build the KNN model meant for learning:
                 - Prepare the data to learn converting to model's requirements.
                 - Fit the model with preprocessed data.
        """
        self.__model.fit_scaler(x_train, x_test)
        x_test  = self.__model.transform(x_test)
        x_train = self.__model.fit_transform(x_train)
        self.__model.fit_model(x_train, y_train)
        return x_train, x_test

    def commit_prediction(self, x_test):
        """
        Stage 3: Performing prediction.
        """
        return self.__model.predict(x_test)

    def build_map(self, x, y, title) -> None:
        """
        Stage 4: Rendering the train/prediction results.
        """
        KNNDashboard(self.__model, x, y, title).render_map()
        return None

    def build_graph(self, x, y, title) -> None:
        """
        Stage 4: Rendering the train/prediction results.
        """
        KNNDashboard(self.__model, x, y, title).render_graph()
        return None

    def estimate(self, x, y) -> float:
        """
        Stage 5: Estimate the model with leave one out method.
        :return: float
        """
        return KNNEstimator(self.__model).leave_one_out(x, y)

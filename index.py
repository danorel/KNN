from src.kit import KNNSciKit
from src.kit import KNNCustom
from src.display import KNNDashboard
from src.dataset import RandomPointDatasetGenerator
from src.dataset import RandomPointDatasetSplitter
from src.estimators import KNNEstimator


class LearningKNNPipeline:
    def __init__(self, Model: KNNCustom or KNNSciKit, size: int, ratio=.8):
        """
        Launches the KNN Model.
        :return: None
        """
        self.__size: int = size
        self.__ratio: float = ratio
        self.__dataset: list or None = None
        self.__Model: KNNCustom or KNNSciKit = Model

    def learn_model(self) -> int:
        max_k = 0
        max_precision = 0

        for k in range(1, 10):
            pipeline: LaunchKNNPipeline = LaunchKNNPipeline(self.__Model, k, self.__size)
            x_train, x_test, y_train, y_test = pipeline.prepare_dataset()
            precision = pipeline.estimate(x_train, y_train)
            print("Precision on neighbours", k, "equals:", precision)
            if precision > max_precision:
                max_precision = precision
                max_k = k

        return max_k


class LaunchKNNPipeline:
    def __init__(self, Model: KNNCustom or KNNSciKit, k: int, size: int, ratio=.8):
        """
        Launches the KNN Model.
        :return: None
        """
        self.__k: int = k
        self.__size: int = size
        self.__ratio: float = ratio
        self.__dataset: list or None = None
        self.__Model: KNNCustom or KNNSciKit = Model(k)

    def prepare_dataset(self):
        """
        Stage 1: Generate dataset with classes.
                 - Generate N points for the whole dataset.
                 - Divide the dataset by train/test.
        """
        self.__dataset = RandomPointDatasetGenerator(self.__size).commit_classified()
        return RandomPointDatasetSplitter(self.__dataset).commit(ratio=self.__ratio)

    def prepare_model(self, x_train, x_test, y_train):
        """
        Stage 2: Build the KNN model meant for learning:
                 - Prepare the data to learn converting to model's requirements.
                 - Fit the model with preprocessed data.
        """
        self.__Model.fit_scaler(x_train, x_test)
        x_test  = self.__Model.transform(x_test)
        x_train = self.__Model.fit_transform(x_train)
        self.__Model.fit_model(x_train, y_train)
        return x_train, x_test

    def commit_prediction(self, x_test):
        """
        Stage 3: Performing prediction.
        """
        return self.__Model.predict(x_test)

    def build_map(self, x, y, title) -> None:
        """
        Stage 4: Rendering the train/prediction results.
        """
        KNNDashboard(self.__Model, x, y, title).render_map()
        return None

    def build_graph(self, x, y, title) -> None:
        """
        Stage 4: Rendering the train/prediction results.
        """
        KNNDashboard(self.__Model, x, y, title).render_graph()
        return None

    def estimate(self, x, y) -> float:
        """
        Stage 5: Estimate the model with leave one out method.
        :return: float
        """
        return KNNEstimator(self.__Model).leave_one_out(x, y)


def launch() -> None:
    pipeline = LaunchKNNPipeline(KNNSciKit, 5, 1000)
    x_train, x_test, y_train, y_test = pipeline.prepare_dataset()
    print("Precision with k neighbours:", pipeline.estimate(x_train, y_train))
    return None


def launch_learning() -> None:
    learning = LearningKNNPipeline(KNNSciKit, 1000)
    best_k = learning.learn_model()
    print("Best k:", best_k)
    return None


if __name__ == '__main__':
    launch()
    launch_learning()

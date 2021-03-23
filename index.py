from src.kit import KNNSciKit
from src.kit import KNNCustom
from src.display import KNNDashboard
from src.dataset import RandomPointDatasetGenerator
from src.dataset import RandomPointDatasetSplitter


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
        x_train_transform, x_test_transform = self.__Model.fit_transform(x_train, x_test)
        self.__Model.fit(x_train_transform, y_train)
        return x_train_transform, x_test_transform

    def commit_prediction(self, x_test):
        """
        Stage 3: Performing prediction.
        """
        return self.__Model.predict(x_test)

    def build_graph(self, x_train, x_test, y_train, y_test) -> None:
        """
        Stage 4: Rendering the train/prediction results.
        """
        KNNDashboard(self.__Model, 'Training set').render(x_train, y_train)
        KNNDashboard(self.__Model, 'Prediction Set').render(x_test, y_test)
        return None


def launch():
    pipeline = LaunchKNNPipeline(KNNSciKit, 5, 100)
    x_train, x_test, y_train, y_test   = pipeline.prepare_dataset()
    x_train_transform, x_test_transform = pipeline.prepare_model(x_train, x_test, y_train)
    y_prediction = pipeline.commit_prediction(x_test)
    pipeline.build_graph(x_train_transform, x_test_transform, y_train, y_prediction)


if __name__ == '__main__':
    launch()

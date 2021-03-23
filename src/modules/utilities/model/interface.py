import abc


class AbstractModelKNN(abc.ABC):
    @abc.abstractmethod
    def prepare_dataset(self):
        """
        Stage 1:
        Generate dataset with classes.
            - Generate N points for the whole dataset.
            - Divide the dataset by train/test.
        """
        pass

    def prepare_model(self, x_train, x_test, y_train):
        """
        Stage 2:
        Build the KNN model meant for learning:
            - Prepare the data to learn converting to model's requirements.
            - Fit the model with preprocessed data.
        """
        pass

    def commit_prediction(self, x_test):
        """
        Stage 3:
        Performing prediction.
        """
        pass

    def build_map(self, x, y, title):
        """
        Stage 4:
        Rendering the train/prediction results.
        """
        pass

    def build_graph(self, x, y, title):
        """
        Stage 4:
        Rendering the train/prediction results.
        """
        pass

    def estimate(self, x, y):
        """
        Stage 5: Estimate the model with leave one out method.
        :return: float
        """
        pass

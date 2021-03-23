import abc


class AbstractKNN(abc.ABC):
    @abc.abstractmethod
    def fit_scaler(self, x_train, y_train):
        pass

    @abc.abstractmethod
    def transform(self, x_test):
        pass

    @abc.abstractmethod
    def fit_transform(self, x_train):
        pass

    @abc.abstractmethod
    def fit_model(self, x_train, y_train):
        pass

    @abc.abstractmethod
    def predict(self, x_test):
        pass

    @abc.abstractmethod
    def get_k(self):
        pass

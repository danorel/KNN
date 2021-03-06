from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from src.modules.kit.interface import AbstractKNN


class KNNEqualCustom(AbstractKNN):
    def __init__(self, k: int):
        self.__k = k
        self.__classifier = KNeighborsClassifier(n_neighbors=k)

    def fit_scaler(self, x_train, y_train):
        pass

    def transform(self, x_test):
        pass

    def fit_transform(self, x_train):
        pass

    def fit_model(self, x_train, y_train):
        pass

    def predict(self, x_test):
        pass

    def get_k(self):
        return self.__k


class KNNEqualSciKit(AbstractKNN):
    def __init__(self, k: int):
        self.__k: int = k
        self.__scaler: StandardScaler = StandardScaler()
        self.__classifier: KNeighborsClassifier = KNeighborsClassifier(n_neighbors=k)

    def fit_scaler(self, x_train, x_test) -> None:
        self.__scaler.fit(x_train, x_test)
        return None

    def fit_transform(self, x_train):
        return self.__scaler.fit_transform(x_train)

    def transform(self, x_test):
        return self.__scaler.transform(x_test)

    def fit_model(self, x_train, y_train) -> None:
        self.__classifier.fit(x_train, y_train)
        return None

    def predict(self, x_test):
        """
        Predicting the test set results.
        :return:
        """
        return self.__classifier.predict(x_test)

    def get_k(self):
        return self.__k

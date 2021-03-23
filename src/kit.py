import abc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


class KNNAbstract(abc.ABC):
    @abc.abstractmethod
    def fit(self, x_train, y_train):
        pass

    @abc.abstractmethod
    def fit_transform(self, x_train, x_test):
        pass

    @abc.abstractmethod
    def predict(self, x_test):
        pass


class KNNCustom(KNNAbstract):
    def __init__(self, k: int):
        super().__init__()
        self.__k = k
        self.__classifier = KNeighborsClassifier(n_neighbors=k)

    def fit(self, x_train, y_train):
        pass

    def fit_transform(self, x_train, x_test):
        pass

    def predict(self, x_test):
        pass


class KNNSciKit(KNNAbstract):
    def __init__(self, k: int):
        super().__init__()
        self.__k: int = k
        self.__scaler: StandardScaler = StandardScaler()
        self.__classifier: KNeighborsClassifier = KNeighborsClassifier(n_neighbors=k)

    def fit_transform(self, x_train, x_test):
        return self.__scaler.fit_transform(x_train), self.__scaler.transform(x_test)

    def fit(self, x_train, y_train):
        self.__classifier.fit(x_train, y_train)
        return

    def predict(self, x_test):
        """
        Predicting the test set results.
        :return:
        """
        return self.__classifier.predict(x_test)

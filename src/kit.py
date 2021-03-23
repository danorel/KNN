import abc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


class KNNAbstract(abc.ABC):
    @abc.abstractmethod
    def fit_transform(self):
        pass

    @abc.abstractmethod
    def predict(self, x_test):
        pass


class KNNCustom(KNNAbstract):
    def __init__(self, k: int):
        super().__init__(self)
        self.__k = k
        self.__classifier = KNeighborsClassifier(n_neighbors=k)

    def fit_transform(self):
        pass

    def predict(self, x_test):
        pass


class KNNSciKit(KNNAbstract):
    def __init__(self, k: int):
        super().__init__(self)
        self.__k = k
        self.__classifier = KNeighborsClassifier(n_neighbors=k)

    @staticmethod
    def fit_transform(x_train, x_test):
        sc = StandardScaler()
        return sc.fit_transform(x_train), sc.transform(x_test)

    def predict(self, x_test):
        """
        Predicting the test set results.
        :return:
        """
        return self.__classifier.predict(x_test)

from sklearn.metrics import confusion_matrix

from src.modules.kit.interface import AbstractKNN
from src.modules.metrics.interface import AbstractEstimatorKNN


class EstimatorKNN(AbstractEstimatorKNN):
    def __init__(self, classifier: AbstractKNN):
        self.__classifier: AbstractKNN = classifier

    def leave_one_out(self, x, y) -> float:
        precision: int = 0
        x_train_rest, y_train_rest = x, y

        while len(x_train_rest) != self.__classifier.get_k():
            # Leave the first element out.
            x_train_rest, x_test_samples = x_train_rest[1:], [x_train_rest[0]]
            y_train_rest, y_test_samples = y_train_rest[1:], [y_train_rest[0]]

            # Fit the train/test samples to the model.
            self.__classifier.fit_scaler(x_train_rest, x_test_samples)
            x_test_transform_sample = self.__classifier.transform(x_test_samples)
            x_train_transform_rest  = self.__classifier.fit_transform(x_train_rest)
            self.__classifier.fit_model(x_train_transform_rest, y_train_rest)

            # Make sample prediction.
            y_sample_prediction = self.__classifier.predict(x_test_transform_sample)

            # Decide if successful prediction. If correct prediction, increase precision.
            if y_test_samples[0] == y_sample_prediction[0]:
                precision += 1

        return precision / len(x)

    def confusion_matrix(self, x, y) -> float:
        cm = confusion_matrix(x, y)
        print(cm)
        return .0

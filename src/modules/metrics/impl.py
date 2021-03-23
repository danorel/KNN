from sklearn.metrics import confusion_matrix


class KNNEstimator:
    def __init__(self, Model):
        """
        Stage 5: Estimate the effectiveness of the model. Leave one out algorithm.
        :return: object
        """
        self.__Model = Model

    def leave_one_out(self, x, y) -> float:
        """
        Percentage of correct-giving results of the model.
        Using leave one out metrics.
        :return:
        """
        precision: int = 0
        x_train_rest, y_train_rest = x, y

        while len(x_train_rest) != self.__Model.get_k():
            # Leave the first element out.
            x_train_rest, x_test_samples = x_train_rest[1:], [x_train_rest[0]]
            y_train_rest, y_test_samples = y_train_rest[1:], [y_train_rest[0]]

            # Fit the train/test samples to the model.
            self.__Model.fit_scaler(x_train_rest, x_test_samples)
            x_test_transform_sample = self.__Model.transform(x_test_samples)
            x_train_transform_rest  = self.__Model.fit_transform(x_train_rest)
            self.__Model.fit_model(x_train_transform_rest, y_train_rest)

            # Make sample prediction.
            y_sample_prediction = self.__Model.predict(x_test_transform_sample)

            # Decide if successful prediction. If correct prediction, increase precision.
            if y_test_samples[0] == y_sample_prediction[0]:
                precision += 1

        return precision / len(x)

    def confusion_matrix(self, x, y) -> float:
        """
        Percentage of correct-giving results of the model.
        Using confusion matrix metrics.
        :return:
        """
        cm = confusion_matrix(x, y)
        print(cm)
        return .0

from src.kit import KNNSciKit
from src.display import KNNDashboard
from src.dataset import RandomPointDatasetGenerator
from src.dataset import RandomPointDatasetSplitter

if __name__ == '__main__':
    dataset = RandomPointDatasetGenerator(100).commit_classified()
    x_train, x_test, y_train, y_test = RandomPointDatasetSplitter(dataset).commit(ratio=.8)

    model = KNNSciKit(10)
    x_train_transform, x_test_transform = model.fit_transform(x_train, x_test)
    model.fit(x_train_transform, y_train)

    y_predict = model.predict(x_test_transform)

    KNNDashboard(model, 'Training set').render(x_train_transform, y_train)
    KNNDashboard(model, 'Prediction Set').render(x_test_transform, y_predict)


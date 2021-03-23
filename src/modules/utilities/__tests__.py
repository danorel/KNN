import unittest

from src.modules.kit.impl.equal import KNNEqualSciKit
from src.modules.utilities.display.impl import DisplayKNN
from src.modules.utilities.learning.impl.equal import LearningKNNEqual
from src.modules.utilities.model.impl.equal import ModelEqualKNN


class KNNEqualModelTestCase(unittest.TestCase):
    def test_knn_5_neighbours_1000_examples(self) -> None:
        pipeline = ModelEqualKNN(KNNEqualSciKit, 5, 1000, 8)
        x_train, x_test, y_train, y_test = pipeline.prepare_dataset()
        self.assertAlmostEqual(pipeline.estimate(x_train, y_train), .8, delta=.1)
        return None


class LearningKNNEqualTestCase(unittest.TestCase):
    def test_learning_model_100_examples(self) -> None:
        learning = LearningKNNEqual(KNNEqualSciKit, 100, 12, 5)
        self.assertAlmostEqual(learning.learn_model(), 1, delta=5)

    def test_learning_model_1000_examples(self) -> None:
        learning = LearningKNNEqual(KNNEqualSciKit, 1000, 12, 5)
        self.assertAlmostEqual(learning.learn_model(), 1, delta=5)
        return None

    def test_learning_model_2500_examples(self) -> None:
        learning = LearningKNNEqual(KNNEqualSciKit, 2500, 12, 5)
        self.assertAlmostEqual(learning.learn_model(), 1, delta=5)
        return None


class DisplayKNNEqualTestCase(unittest.TestCase):
    def test_display_knn_equal_5_neighbours_1000_examples(self) -> None:
        pipeline = ModelEqualKNN(KNNEqualSciKit, 5, 1000, 8)
        x_train, x_test, y_train, y_test = pipeline.prepare_dataset()
        x_train, x_test = pipeline.prepare_model(x_train, x_test, y_train)
        pipeline.build_map(x_train, y_train, 'Training Set')
        self.assertTrue(True)
        return None


if __name__ == '__main__':
    unittest.main()

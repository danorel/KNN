import unittest

from src.modules.kit.impl.equal import KNNEqualSciKit
from src.modules.utilities.learning import LearningKNNEqualPipeline
from src.modules.utilities.configuration import KNNEqualPipeline


class KitKNNExampleTest(unittest.TestCase):
    def test_knn_5_neighbours_1000_examples(self) -> None:
        pipeline = KNNEqualPipeline(KNNEqualSciKit, 5, 1000, 8)
        x_train, x_test, y_train, y_test = pipeline.prepare_dataset()
        self.assertAlmostEqual(pipeline.estimate(x_train, y_train), .8, delta=.1)
        return None


class KitKNNLearningTest(unittest.TestCase):
    def test_learning_model_100_examples(self):
        learning = LearningKNNEqualPipeline(KNNEqualSciKit, 100, 12, 5)
        best_k = learning.learn_model()
        self.assertAlmostEqual(best_k, 1, delta=5)

    def test_learning_model_1000_examples(self):
        learning = LearningKNNEqualPipeline(KNNEqualSciKit, 1000, 12, 5)
        best_k = learning.learn_model()
        self.assertAlmostEqual(best_k, 1, delta=5)

    def test_learning_model_2500_examples(self):
        learning = LearningKNNEqualPipeline(KNNEqualSciKit, 2500, 12, 5)
        best_k = learning.learn_model()
        self.assertAlmostEqual(best_k, 1, delta=5)


if __name__ == '__main__':
    unittest.main()

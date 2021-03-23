import unittest

from src.modules.kit.equal import KNNSciKit
from src.modules.pipelines.learning import LearningKNNPipeline


class KitKNNLearningTest(unittest.TestCase):
    def test_learning_model_100_examples(self):
        learning = LearningKNNPipeline(KNNSciKit, 100)
        best_k = learning.learn_model()
        self.assertEqual(best_k, 1)

    def test_learning_model_1000_examples(self):
        learning = LearningKNNPipeline(KNNSciKit, 1000)
        best_k = learning.learn_model()
        self.assertEqual(best_k, 1)

    def test_learning_model_2500_examples(self):
        learning = LearningKNNPipeline(KNNSciKit, 2500)
        best_k = learning.learn_model()
        self.assertEqual(best_k, 1)


if __name__ == '__main__':
    unittest.main()

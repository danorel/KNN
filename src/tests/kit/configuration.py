import unittest

from src.modules.kit.equal import KNNSciKit
from src.modules.pipelines.configuration import LaunchKNNPipeline


class KitKNNExampleTest(unittest.TestCase):
    def test_knn_5_neighbours_1000_examples(self) -> None:
        pipeline = LaunchKNNPipeline(KNNSciKit, 5, 1000)
        x_train, x_test, y_train, y_test = pipeline.prepare_dataset()
        self.assertAlmostEqual(pipeline.estimate(x_train, y_train), .8, .1)
        return None


if __name__ == '__main__':
    unittest.main()

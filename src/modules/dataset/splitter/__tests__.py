import unittest

from src.modules.dataset.classifier.impl import PointClassifier
from src.modules.dataset.generator.impl import RandomListGenerator
from src.modules.dataset.generator.impl import RandomValueGenerator
from src.modules.dataset.generator.impl import RandomPointDatasetGenerator
from src.modules.dataset.splitter.impl import RandomPointDatasetSplitter


class DatabaseSplitterTestCase(unittest.TestCase):
    def test_point_database_splitter_1000_samples(self):
        x_train, x_test, y_train, y_test = RandomPointDatasetSplitter(
            dataset=RandomPointDatasetGenerator(
                generator=RandomListGenerator(generator=RandomValueGenerator()),
                classifier=PointClassifier(12)).commit_classified(1000)
        ).commit(.7)
        self.assertAlmostEqual(700, len(x_train), delta=25)
        self.assertAlmostEqual(700, len(y_train), delta=25)
        self.assertAlmostEqual(300, len(x_test), delta=25)
        self.assertAlmostEqual(300, len(y_test), delta=25)

    def test_point_database_splitter_10000_samples(self):
        x_train, x_test, y_train, y_test = RandomPointDatasetSplitter(
            dataset=RandomPointDatasetGenerator(
                generator=RandomListGenerator(generator=RandomValueGenerator()),
                classifier=PointClassifier(12)).commit_classified(10000)
        ).commit(.7)
        self.assertAlmostEqual(7000, len(x_train), delta=100)
        self.assertAlmostEqual(7000, len(y_train), delta=100)
        self.assertAlmostEqual(3000, len(x_test), delta=100)
        self.assertAlmostEqual(3000, len(y_test), delta=100)


if __name__ == '__main__':
    unittest.main()

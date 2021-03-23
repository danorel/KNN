import unittest

from src.modules.dataset.classifier.impl import PointClassifier
from src.modules.dataset.generator.impl import RandomValueGenerator
from src.modules.dataset.generator.impl import RandomListGenerator
from src.modules.dataset.generator.impl import RandomPointDatasetGenerator


class GeneratorTestCase(unittest.TestCase):
    def test_value_generator_from_minus_1_to_1(self) -> None:
        self.assertAlmostEqual(0, RandomValueGenerator().rand(), delta=1)
        return None

    def test_value_generator_from_minus_5_to_5(self) -> None:
        self.assertAlmostEqual(0, RandomValueGenerator(-5, 5).rand(), delta=5)
        return None

    def test_list_generator_from_minus_1_to_1(self) -> None:
        for value in RandomListGenerator(generator=RandomValueGenerator()).rand(10):
            self.assertAlmostEqual(0, value, delta=1)
            self.assertLess(value, 1, 'Cannot be greater than 1!')
            self.assertGreater(value, -1, 'Cannot be less than -1!')
        return None

    def test_list_generator_from_minus_5_to_5(self) -> None:
        for value in RandomListGenerator(generator=RandomValueGenerator(-5, 5)).rand(10):
            self.assertAlmostEqual(0, value, delta=5)
            self.assertLess(value, 5, 'Cannot be greater than 5!')
            self.assertGreater(value, -5, 'Cannot be less than -5!')
        return None

    def test_point_dataset_generator_from_minus_1_to_1(self) -> None:
        for point in RandomPointDatasetGenerator(
                generator=RandomListGenerator(generator=RandomValueGenerator()),
                classifier=PointClassifier(12)).commit_classified(1000):
            self.assertAlmostEqual(0, point[0][0], delta=1)
            self.assertAlmostEqual(0, point[0][1], delta=1)
            self.assertLess(point[0][0], 1, 'Point X coordinate cannot be greater than 1!')
            self.assertGreater(point[0][0], -1, 'Point X coordinate cannot be less than -1!')
            self.assertLess(point[0][1], 1, 'Point Y coordinate cannot be greater than 1!')
            self.assertGreater(point[0][1], -1, 'Point Y coordinate cannot be less than -1!')
        return None

    def test_point_dataset_generator_from_minus_5_to_5(self) -> None:
        for point in RandomPointDatasetGenerator(
                generator=RandomListGenerator(generator=RandomValueGenerator(-5, 5)),
                classifier=PointClassifier(12)).commit_classified(1000):
            self.assertAlmostEqual(0, point[0][0], delta=5)
            self.assertAlmostEqual(0, point[0][1], delta=5)
            self.assertLess(point[0][0], 5, 'Point X coordinate cannot be greater than 5!')
            self.assertGreater(point[0][0], -5, 'Point X coordinate cannot be less than -5!')
            self.assertLess(point[0][1], 5, 'Point Y coordinate cannot be greater than 5!')
            self.assertGreater(point[0][1], -5, 'Point Y coordinate cannot be less than -5!')
        return None


if __name__ == '__main__':
    unittest.main()

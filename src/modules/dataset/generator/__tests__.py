import unittest

from src.modules.dataset.generator.impl import RandomValueGenerator
from src.modules.dataset.generator.impl import RandomListGenerator


class GeneratorTestCase(unittest.TestCase):
    def test_value_generator_from_minus_1_to_1(self):
        self.assertAlmostEqual(0, RandomValueGenerator().rand(), delta=1)

    def test_value_generator_from_minus_5_to_5(self):
        self.assertAlmostEqual(0, RandomValueGenerator(-5, 5).rand(), delta=5)

    def test_list_generator_from_minus_1_to_1(self):
        for value in RandomListGenerator(generator=RandomValueGenerator()).rand(10):
            self.assertAlmostEqual(0, value, delta=1)
            self.assertLess(value, 1, 'Cannot be greater than 1!')
            self.assertGreater(value, -1, 'Cannot be less than -1!')

    def test_list_generator_from_minus_5_to_5(self):
        for value in RandomListGenerator(generator=RandomValueGenerator(-5, 5)).rand(10):
            self.assertAlmostEqual(0, value, delta=5)
            self.assertLess(value, 5, 'Cannot be greater than 5!')
            self.assertGreater(value, -5, 'Cannot be less than -5!')


if __name__ == '__main__':
    unittest.main()

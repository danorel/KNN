import unittest

from src.entities.structures import Point
from src.modules.dataset.classifier.impl import PointClassifier


class ClassifierTestCase(unittest.TestCase):
    def test_point_classifier_plane_1_class_1(self):
        point: Point = (0.25, 0.25)
        self.assertEqual(1, PointClassifier(12).commit(point))

    def test_point_classifier_plane_1_class_2(self):
        point: Point = (0.5, 0.5)
        self.assertEqual(2, PointClassifier(12).commit(point))

    def test_point_classifier_plane_1_class_3(self):
        point: Point = (0.86, 0.86)
        self.assertEqual(3, PointClassifier(12).commit(point))

    def test_point_classifier_plane_3_class_9(self):
        point: Point = (-0.5, -0.5)
        self.assertEqual(9, PointClassifier(12).commit(point))


if __name__ == '__main__':
    unittest.main()

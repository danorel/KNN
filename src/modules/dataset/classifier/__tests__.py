import unittest

from src.entities.structures import Point
from src.modules.dataset.classifier.impl import PointClassifier


class ClassifierTestCase(unittest.TestCase):
    def test_point_classifier_plane_1_class_1(self) -> None:
        point: Point = (0.25, 0.25)
        self.assertEqual(2, PointClassifier(12).commit(point))
        return None

    def test_point_classifier_plane_1_class_2(self) -> None:
        point: Point = (0.5, 0.5)
        self.assertEqual(2, PointClassifier(12).commit(point))
        return None

    def test_point_classifier_plane_1_class_3(self) -> None:
        point: Point = (0.16, 0.86)
        self.assertEqual(3, PointClassifier(12).commit(point))
        return None

    def test_point_classifier_plane_2_class_4(self) -> None:
        point: Point = (-0.15, 0.8)
        self.assertEqual(4, PointClassifier(12).commit(point))
        return None

    def test_point_classifier_plane_3_class_9(self) -> None:
        point: Point = (-0.5, -0.5)
        self.assertEqual(8, PointClassifier(12).commit(point))
        return None

    def test_point_classifier_plane_3_class_9(self) -> None:
        point: Point = (-0.25, -0.8)
        self.assertEqual(9, PointClassifier(12).commit(point))
        return None

    def test_point_classifier_plane_4_class_10(self) -> None:
        point: Point = (0.15, -0.8)
        self.assertEqual(10, PointClassifier(12).commit(point))
        return None

    def test_point_classifier_plane_4_class_11(self) -> None:
        point: Point = (0.5, -0.5)
        self.assertEqual(11, PointClassifier(12).commit(point))
        return None

    def test_point_classifier_plane_4_class_12(self) -> None:
        point: Point = (0.85, -0.15)
        self.assertEqual(12, PointClassifier(12).commit(point))
        return None


if __name__ == '__main__':
    unittest.main()

import unittest
from utils import are_overlapping, generate_background_bbox


class TestUtils(unittest.TestCase):
    def test_are_overlapping(self):
        # Bounding box is represented by a list
        # in the form [ymin, xmin, ymax, xmax]
        self.assertTrue(are_overlapping([2, 1, 7, 6], [5, 4, 11, 8]))
        self.assertFalse(are_overlapping([2, 1, 7, 6], [8, 6, 13, 9]))

    def test_generate_background(self):
        existing_bboxes = [[3, 4, 8, 9], [13, 1, 16, 5]]
        bbox_shape = (5, 5)
        image_shape = (23, 10)

        def test():
            try:
                bbox = generate_background_bbox(
                    image_shape,
                    bbox_shape,
                    existing_bboxes
                )
            except RuntimeError:
                return

            for existing_bbox in existing_bboxes:
                self.assertFalse(are_overlapping(bbox, existing_bbox))

        for i in range(10):
            test()


if __name__ == "__main__":
    unittest.main()

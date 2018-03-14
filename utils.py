from numpy.random import randint


roaddamage_label_names = (
    'D00', 'D01', 'D10',
    'D11', 'D20', 'D30',
    'D40', 'D43', 'D44'
)


def are_overlapping(bbox1, bbox2):
    ymin1, xmin1, ymax1, xmax1 = bbox1
    ymin2, xmin2, ymax2, xmax2 = bbox2

    vertexes = [
        (xmin1, ymin1),
        (xmin1, ymax1),
        (xmax1, ymin1),
        (xmax1, ymax1)
    ]

    for (x, y) in vertexes:
        if xmin2 < x < xmax2 and ymin2 < y < ymax2:
            return True
    return False


def generate_background_bbox(image_shape, bbox_shape, existing_bboxes,
                             n_attempts=10):
    """
    Generate a bounding box that does not overlap with any `existing_bboxes`.
    The function tries generating a bounding box at most `n_attempts` times.
    Raises `RuntimeError` if a bounding box that doesn't overlap with any
    existing bounding boxes cannot be generated.

    Args:
        image_shape (tuple): The shape of the original image in the format of
            (height, width). Bounding boxes are generated to fit within
            `image_shape`.
        bbox_shape (tuple): The shape of a bounding box to be generated.
        existing_bboxes (list of tuples): Existing bounding boxes. The
            generated bounding box should not overlap with any
            `existing_bboxes`.
        n_attempts (int): The number of attempts to generate a bounding box
    """

    def generate_candidate():
        xmin = randint(0, image_shape[0] - bbox_shape[0] + 1)
        ymin = randint(0, image_shape[1] - bbox_shape[1] + 1)
        xmax = xmin + bbox_shape[0]
        ymax = ymin + bbox_shape[1]
        return (ymin, xmin, ymax, xmax)

    def at_least_one_overlapping(candidate, bboxes):
        """
        Whether there is at least one bbox that overlaps with the candidate
        """
        for bbox in bboxes:
            if are_overlapping(candidate, bbox):
                return True
        return False

    for i in range(n_attempts):
        candidate = generate_candidate()
        if at_least_one_overlapping(candidate, existing_bboxes):
            continue
        # return if there is no existing bounding box
        # that overlaps with the candidate
        return candidate

    raise RuntimeError("Background could not be generated")

def load_labels_and_bboxes(dataset, indices=None):
    if indices is None:
        indices = range(len(dataset))

    bboxes = []
    labels = []
    for i in indices:
        img, bbox, label = dataset.get_example(i)
        if len(bbox) == 0:
            continue
        labels.append(label)
        bboxes.append(bbox)
    return labels, bboxes

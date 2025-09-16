def get_center_of_bbox(bbox):
    """
    Calculate the center point of a bounding box.

    Parameters:
    bbox (list or tuple): A list or tuple containing the coordinates of the bounding box in the format [x1, y1, x2, y2].

    Returns:
    tuple: A tuple containing the (x, y) coordinates of the center point of the bounding box.
    """
    x1, y1, x2, y2 = bbox
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    return (center_x, center_y)

def measure_distance(p1, p2):
    """
    Calculate the Euclidean distance between two points.

    Parameters:
    p1 (tuple): A tuple containing the (x, y) coordinates of the first point.
    p2 (tuple): A tuple containing the (x, y) coordinates of the second point.

    Returns:
    float: The Euclidean distance between the two points.
    """
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5
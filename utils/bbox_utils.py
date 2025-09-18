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

def get_foot_position(bbox):
    """
    Get the foot position (bottom center) of a bounding box.

    Parameters:
    bbox (list or tuple): A list or tuple containing the coordinates of the bounding box in the format [x1, y1, x2, y2].

    Returns:
    tuple: A tuple containing the (x, y) coordinates of the foot position of the bounding box.
    """
    x1, y1, x2, y2 = bbox
    foot_x = int((x1 + x2) / 2)
    foot_y = int(y2)
    return (foot_x, foot_y)

def get_closest_key_point_index(point, keypoints, keypoint_indices):
    """
    Get the index of the closest key point to a given point from a list of candidate indices.

    Parameters:
    point (tuple): A tuple containing the (x, y) coordinates of the reference point.
    key_points (list): A list of tuples containing the (x, y) coordinates of key points.
    candidate_indices (list): A list of indices representing the candidate key points to consider.

    Returns:
    int: The index of the closest key point from the candidate indices.
    """
    closest_distance = float('inf')
    key_point_ind = keypoint_indices[0]
    for keypoint_index in keypoint_indices:
        keypoint = keypoints[keypoint_index*2], keypoints[keypoint_index*2 + 1]
        distance = abs(point[1] - keypoint[1])  # Only consider vertical distance (y-axis)

        if distance < closest_distance:
            closest_distance = distance
            key_point_ind = keypoint_index
        
    return key_point_ind


def get_height_of_bbox(bbox):
    """
    Calculate the height of a bounding box.

    Parameters:
    bbox (list or tuple): A list or tuple containing the coordinates of the bounding box in the format [x1, y1, x2, y2].

    Returns:
    int: The height of the bounding box.
    """
    return bbox[3] - bbox[1]

def measure_xy_distance(p1, p2):
    """
    Calculate the horizontal and vertical distances between two points.

    Parameters:
    p1 (tuple): A tuple containing the (x, y) coordinates of the first point.
    p2 (tuple): A tuple containing the (x, y) coordinates of the second point.

    Returns:
    tuple: A tuple containing the horizontal (delta_x) and vertical (delta_y) distances between the two points.
    """
    return abs(p1[0] - p2[0]), abs(p1[1] - p2[1])
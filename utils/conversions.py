
def convert_pixel_distance_to_meters(pixel_distance, reference_height_in_meters, reference_height_in_pixels):
    # Convert a distance in pixels to meters using a reference height.
    return (pixel_distance * reference_height_in_meters) / reference_height_in_pixels

def convert_meters_to_pixel_distance(meters, reference_height_in_meters, reference_height_in_pixels):
    # Convert a distance in meters to pixels using a reference height.
    return (meters * reference_height_in_pixels) / reference_height_in_meters
import cv2
import sys
import numpy as np
sys.path.append('../')
import constants
from utils import(convert_pixel_distance_to_meters, 
                  convert_meters_to_pixel_distance,
                  get_foot_position,
                  get_closest_key_point_index,
                  get_height_of_bbox,
                  measure_xy_distance,
                  get_center_of_bbox,
                  measure_distance)

class MiniCourt:
    def __init__(self, frame):
        self.drawing_rectangle_width = 250
        self.drawing_rectangle_height = 500
        self.buffer = 50
        self.padding_court = 20

        self.set_canvas_background_box_position(frame)
        self.set_mini_court_position()
        self.set_court_drawing_key_points()
        self.set_court_lines()

    def convert_meters_to_pixels(self, meters):
        return convert_meters_to_pixel_distance(meters,
                                                    constants.COURT_WIDTH,
                                                    self.court_drawing_width
                                                )


    def set_court_drawing_key_points(self):
        drawing_key_points = [0] * 24

        # Adjustmets to fit court onto correct parts of the image
        dx = 0
        dy = 0

        # --- corners (pickleball order) ---
        x_left  = int(self.court_start_x) + dx
        x_right = int(self.court_end_x) + dx
        y_top   = int(self.court_start_y) + dy
        y_bot   = int(self.court_start_y + self.convert_meters_to_pixels(constants.HALF_COURT_LENGTH * 2)) + dy

        # 0: bottom-left
        drawing_key_points[0], drawing_key_points[1] = x_left,  y_bot
        # 1: top-left
        drawing_key_points[2], drawing_key_points[3] = x_left,  y_top
        # 2: top-right
        drawing_key_points[4], drawing_key_points[5] = x_right, y_top
        # 3: bottom-right
        drawing_key_points[6], drawing_key_points[7] = x_right, y_bot

        # --- helpers ---
        x_mid = (x_left + x_right) // 2
        y_net = (y_top + y_bot) // 2
        nvz_px = int(self.convert_meters_to_pixels(constants.KITCHEN_LENGTH))  # NVZ/kitchen depth in px
        y_close = y_net + nvz_px   # toward bottom
        y_far   = y_net - nvz_px   # toward top

        # 4: close middle left
        drawing_key_points[8],  drawing_key_points[9]  = x_left,  y_close
        # 5: close middle right
        drawing_key_points[10], drawing_key_points[11] = x_right, y_close
        # 6: far middle right
        drawing_key_points[12], drawing_key_points[13] = x_right, y_far
        # 7: far middle left
        drawing_key_points[14], drawing_key_points[15] = x_left,  y_far
        # 8: close middle middle
        drawing_key_points[16], drawing_key_points[17] = x_mid,   y_close
        # 9: far middle middle
        drawing_key_points[18], drawing_key_points[19] = x_mid,   y_far
        # 10: top middle
        drawing_key_points[20], drawing_key_points[21] = x_mid,   y_top
        # 11: bottom middle
        drawing_key_points[22], drawing_key_points[23] = x_mid,   y_bot

        self.drawing_key_points = drawing_key_points

    def set_court_lines(self):
        self.lines = [
            # Court boundary (rectangle)
            (0, 1),  # left sideline
            (1, 2),  # top baseline
            (2, 3),  # right sideline
            (3, 0),  # bottom baseline
            # Centerline (splits left/right service boxes)
            (10, 11),
            # Non-volley zone (kitchen) lines – parallel to net
            (4, 5),  # near-side NVZ line (close to bottom)
            (7, 6),  # far-side  NVZ line (toward top)
        ]



    def set_mini_court_position(self):
        self.court_start_x = self.start_x + self.padding_court
        self.court_start_y = self.start_y + self.padding_court
        self.court_end_x = self.end_x - self.padding_court
        self.court_end_y = self.end_y - self.padding_court
        self.court_drawing_width = self.court_end_x - self.court_start_x

    def set_canvas_background_box_position(self, frame):
        frame = frame.copy()

        self.end_x = frame.shape[1] - self.buffer
        self.end_y = self.buffer + self.drawing_rectangle_height
        self.start_x = self.end_x - self.drawing_rectangle_width
        self.start_y = self.end_y - self.drawing_rectangle_height

    def draw_court(self, frame):
        # Draw key points
        for i in range(0, len(self.drawing_key_points), 2):
            x = int(self.drawing_key_points[i])
            y = int(self.drawing_key_points[i + 1])
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

        # Draw court lines
        for line in self.lines:
            start_point = (int(self.drawing_key_points[line[0]*2]), int(self.drawing_key_points[line[0]*2 + 1]))
            end_point = (int(self.drawing_key_points[line[1]*2]), int(self.drawing_key_points[line[1]*2 + 1]))
            cv2.line(frame, start_point, end_point, (0, 0, 0), 2)

        # Draw net line
        x_left  = int(self.drawing_key_points[0])   # pt 0 x
        x_right = int(self.drawing_key_points[6])   # pt 3 x
        y_net   = int((self.drawing_key_points[21] + self.drawing_key_points[23]) // 2)  # mean of pts 10y & 11y

        cv2.line(frame, (x_left, y_net), (x_right, y_net), (255, 0, 0), 2)

        return frame

    def draw_background_rectangle(self, frame):
        shapes = np.zeros_like(frame, np.uint8)
        # Draw filled rectangle with transparent grey
        cv2.rectangle(shapes, (self.start_x, self.start_y), (self.end_x, self.end_y), (255, 255, 255), cv2.FILLED)
        out = frame.copy()
        alpha = 0.5
        mask = shapes.astype(bool)
        out[mask] = cv2.addWeighted(frame, alpha, shapes, 1 - alpha, 0)[mask]

        return out
    
    def draw_mini_court(self, frames):
        output_frames = []
        for frame in frames:
            frame = self.draw_background_rectangle(frame)
            frame = self.draw_court(frame)
            output_frames.append(frame)

        return output_frames


    def get_start_point_of_mini_court(self):
        return (self.court_start_x, self.court_start_y)
    
    def get_width_of_mini_court(self):
        return self.court_drawing_width
    
    def get_court_drawing_keypoints(self):
        return self.drawing_key_points
    

    # --- COMMENT OUT(WIP) ---
    
    def get_mini_court_coordinates(self, 
                                   object_position, 
                                   closest_key_point, 
                                   closes_key_point_index, 
                                   player_height_in_pixels, 
                                   player_height_in_meters):
        distance_from_x_keypoint_pixels, distance_from_y_keypoint_pixels = measure_xy_distance(object_position, closest_key_point)
        
        # Convert pixel distances to meters
        distance_from_x_keypoint_meters = convert_pixel_distance_to_meters(distance_from_x_keypoint_pixels,
                                                                           player_height_in_meters,
                                                                           player_height_in_pixels)
        distance_from_y_keypoint_meters = convert_pixel_distance_to_meters(distance_from_y_keypoint_pixels,
                                                                           player_height_in_meters,
                                                                           player_height_in_pixels)
        
        # Convert to mini court pixel coordinates
        mini_court_x_distance_pixels = self.convert_meters_to_pixels(distance_from_x_keypoint_meters)
        mini_court_y_distance_pixels = self.convert_meters_to_pixels(distance_from_y_keypoint_meters)
        closest_mini_court_keypoint = (self.drawing_key_points[closes_key_point_index*2],
                                       self.drawing_key_points[closes_key_point_index*2 + 1])
        mini_court_player_position = (closest_mini_court_keypoint[0] + mini_court_x_distance_pixels,
                                      closest_mini_court_keypoint[1] + mini_court_y_distance_pixels)


    def convert_bounding_boxes_to_mini_court_coordinates(self, player_boxes, ball_boxes, original_court_key_points):
        player_heights = {
            1: constants.PLAYER_1_HEIGHT_METERS,
            2: constants.PLAYER_2_HEIGHT_METERS
            # 3: constants.PLAYER_3_HEIGHT_METERS,
            # 4: constants.PLAYER_4_HEIGHT_METERS
        }

        output_player_boxes = []
        output_ball_boxes = []

        for frame_num, player_bbox in enumerate(player_boxes):
            ball_bbox = ball_boxes[frame_num][1]
            ball_position = get_center_of_bbox(ball_bbox)
            closest_player_id_to_ball = min(player_bbox.keys(), key=lambda x: measure_distance(ball_position, get_foot_position(player_bbox[x])))

            output_player_bboxes_dict = {}
            for player_id, bbox in player_bbox.items():
                foot_position = get_foot_position(bbox)

                # Get the closest key point on the court to the foot position - using points(0-close, 1-far, 8-close_middle, 9-far_middle)
                closest_key_point_index = get_closest_key_point_index(foot_position, original_court_key_points, [0, 1, 8, 9])
                closest_key_point = (original_court_key_points[closest_key_point_index*2], 
                                     original_court_key_points[closest_key_point_index*2 + 1])

                # Get player height in pixels
                frame_index_min = max(0, frame_num - 20)
                frame_index_max = min(len(player_boxes), frame_num + 50)
                bbox_heights_in_pixels = [get_height_of_bbox(player_boxes[i][player_id]) for i in range(frame_index_min, frame_index_max)]
                max_player_height_in_pixels = max(bbox_heights_in_pixels)

                mini_court_player_position = self.get_mini_court_coordinates(foot_position,
                                                                            closest_key_point,
                                                                            closest_key_point_index,
                                                                            max_player_height_in_pixels,
                                                                            player_heights[player_id])
                
                output_player_bboxes_dict[player_id] = mini_court_player_position

                if closest_player_id_to_ball == player_id:
                   # Get the closest key point on the court to the foot position - using points(0-close, 1-far, 8-close_middle, 9-far_middle)
                    closest_key_point_index = get_closest_key_point_index(ball_position, original_court_key_points, [0, 1, 8, 9])
                    closest_key_point = (original_court_key_points[closest_key_point_index*2], 
                                        original_court_key_points[closest_key_point_index*2 + 1])
                    mini_court_player_position = self.get_mini_court_coordinates(ball_position,
                                                                            closest_key_point,
                                                                            closest_key_point_index,
                                                                            max_player_height_in_pixels,
                                                                            player_heights[player_id])
                    
                    output_ball_boxes.append({1:mini_court_player_position})
            output_player_boxes.append(output_player_bboxes_dict)

        return output_player_boxes, output_ball_boxes


    def draw_points_on_mini_court(self, frames, positions, color = (0, 255, 0)):
        for frame_num, frame in enumerate(frames):
            for _, position in positions[frame_num].items():
                x, y = position
                x = int(x)
                y = int(y)
                cv2.circle(frame, (x, y), 5, color, -1)

        return frames
    
    # --- COMMENT OUT(END WIP) ---

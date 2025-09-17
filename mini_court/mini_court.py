import cv2
import sys
import numpy as np
sys.path.append('../')
import constants
from utils import(convert_pixel_distance_to_meters, 
                  convert_meters_to_pixel_distance)

class MiniCourt:
    def __init__(self, frame):
        self.drawing_rectangle_width = 250
        self.drawing_rectangle_height = 450
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

        # --- corners (pickleball order) ---
        x_left  = int(self.court_start_x)
        x_right = int(self.court_end_x)
        y_top   = int(self.court_start_y)
        y_bot   = int(self.court_start_y + self.convert_meters_to_pixels(constants.HALF_COURT_LENGTH * 2))

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
        for i in range(0, len(self.drawing_key_points), 2):
            x = int(self.drawing_key_points[i])
            y = int(self.drawing_key_points[i + 1])
            cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)

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
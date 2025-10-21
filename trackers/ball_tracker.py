from ultralytics import YOLO
from math import hypot
import cv2
import pickle
import pandas as pd

class BallTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.prev_center  = None
        self.prev_box     = None
        self.vel          = (0.0, 0.0)
        self.misses       = 0
        self.area_avg     = None
        self.switch_votes = 0
        names_lc = {i: n.lower() for i, n in self.model.names.items()}
        self.ball_cls = next((i for i, n in names_lc.items()
                            if n in ("pickleball", "sports ball", "ball")), None)
        print("ball class idx =", self.ball_cls, "| classes =", self.model.names)


    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1, []) for x in ball_positions]
        # Convert the list into pandas DataFrame for easier manipulation
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        # Interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        # Convert back to list of dictionaries
        ball_positions = [{1:x} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions

    def get_ball_shot_frames(self, ball_positions):
        ball_positions = [x.get(1, []) for x in ball_positions]
        # Convert the list into pandas DataFrame for easier manipulation
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        df_ball_positions['ball_hit'] = 0
        df_ball_positions['mid_y'] = (df_ball_positions['y1'] + df_ball_positions['y2']) / 2
        df_ball_positions['mid_y_rolling_mean'] = df_ball_positions['mid_y'].rolling(window = 5, min_periods = 1, center = False).mean()
        df_ball_positions['delta_y'] = df_ball_positions['mid_y_rolling_mean'].diff()
        minimum_change_frames_for_hit = 18 # Change sensitivity for hit detection
        for i in range(1, len(df_ball_positions) - int(minimum_change_frames_for_hit * 1.2)):
            negative_position_change = df_ball_positions['delta_y'].iloc[i] > 0 and df_ball_positions['delta_y'].iloc[i + 1] < 0 
            positive_position_change = df_ball_positions['delta_y'].iloc[i] < 0 and df_ball_positions['delta_y'].iloc[i + 1] > 0

            if negative_position_change or positive_position_change:
                change_count = 0
                for change_frame in range(i + 1, i + int(minimum_change_frames_for_hit * 1.2)+1):
                    negative_position_change_following_frame = df_ball_positions['delta_y'].iloc[i] > 0 and df_ball_positions['delta_y'].iloc[change_frame] < 0 
                    positive_position_change_following_frame = df_ball_positions['delta_y'].iloc[i] < 0 and df_ball_positions['delta_y'].iloc[change_frame] > 0

                    if negative_position_change and negative_position_change_following_frame:
                        change_count += 1
                    elif positive_position_change and positive_position_change_following_frame:
                        change_count += 1

                if change_count > minimum_change_frames_for_hit-1:
                    df_ball_positions['ball_hit'].iloc[i] = 1

        frame_nums_with_ball_hits = df_ball_positions[df_ball_positions['ball_hit'] == 1].index.tolist()
        return frame_nums_with_ball_hits


    def detect_frames(self, frames, read_from_stub = False, stub_path = None):
        ball_detections = []

        # If reading from stub, load detections from pickle file
        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                ball_detections = pickle.load(f)
            return ball_detections

        for frame in frames:
            ball_dict = self.detect_frame(frame)
            ball_detections.append(ball_dict)

        # Save detections to stub file if path is provided
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(ball_detections, f)

        return ball_detections

    def detect_frame(self, frame, person_boxes = None):
        # results = self.model.predict(frame, conf = 0.15)[0]
    
        # ball_dict = {}
        # for box in results.boxes:
        #     result = box.xyxy.tolist()[0]            
        #     ball_dict[1] = result

        # return ball_dict

        # -----------------------------------------------
        # --- Potential fix for tracking ball issues ---
        # -----------------------------------------------
        if person_boxes is None:
            person_boxes = []
        # 1) predict (same API, but pass ball class if we found it and use a bigger input)
        results = self.model.predict(
            frame,
            conf=0.15,                          # tune 0.12–0.22 as needed
            iou=0.30,
            imgsz=1280,                         # helps tiny balls
            classes=[self.ball_cls] if self.ball_cls is not None else None)[0]

        boxes = results.boxes
        ball_dict = {}
        if boxes is None or len(boxes) == 0:
            return ball_dict  # nothing this frame

        h, w = frame.shape[:2]

        # 2) build candidates; keep small & roughly round-ish; drop anything inside a player box
        def center(b): 
            x1,y1,x2,y2 = b; return (0.5*(x1+x2), 0.5*(y1+y1))  # <- y1+y2, typo guard below

        def _inside(cx, cy, pb):
            px1,py1,px2,py2 = pb
            return (px1 <= cx <= px2) and (py1 <= cy <= py2)

        cands = []
        for b in boxes:
            xyxy = b.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = xyxy
            bw, bh = x2 - x1, y2 - y1
            if bw <= 0 or bh <= 0:
                continue
            area = bw * bh
            ar   = bw / max(bh, 1e-6)
            # size/shape gate: reject net patches, keep plausible balls
            if area > 0.022 * w * h or not (0.7 <= ar <= 1.6):
                continue
            cx, cy = (0.5*(x1+x2), 0.5*(y1+y2))  # correct center
            # reject if center lies inside any player box (kills sleeve hits)
            if person_boxes and any(_inside(cx, cy, pb) for pb in person_boxes):
                continue
            cands.append((float(b.conf[0]), xyxy))

        if not cands:
            return ball_dict

        # 3) choose candidate
        if self.prev_center is not None:
            pcx, pcy = self.prev_center
            # nearest to previous center; tie-break by higher conf
            conf, pick = min(cands, key=lambda t: ((0.5*(t[1][0]+t[1][2]) - pcx)**2 + (0.5*(t[1][1]+t[1][3]) - pcy)**2, -t[0]))[0:2]
        else:
            # first hit: highest confidence
            conf, pick = max(cands, key=lambda t: t[0])

        self.prev_center = (0.5*(pick[0]+pick[2]), 0.5*(pick[1]+pick[3]))
        ball_dict[1] = pick.tolist()
        return ball_dict
        # -----------------------------
        # --- End of potential fix ---
        # -----------------------------

    
    def draw_bboxes(self, video_frames, player_detections):
        output_video_frames = []
        for frame, ball_dict in zip(video_frames, player_detections):
            # Draw ball bounding boxes
            for track_id, bbox in ball_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f"Ball ID: {track_id}", (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 225, 255), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
            output_video_frames.append(frame)

        return output_video_frames
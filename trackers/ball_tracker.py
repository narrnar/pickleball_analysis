from ultralytics import YOLO
import cv2
import pickle
import pandas as pd

class BallTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

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

    def detect_frame(self, frame):
        # results = self.model.predict(frame, conf = 0.15)[0]
    
        # ball_dict = {}
        # for box in results.boxes:
        #     result = box.xyxy.tolist()[0]            
        #     ball_dict[1] = result

        # return ball_dict

        # --- Potential fix for tracking ball issues ---

        results = self.model.predict(frame, 
                                     conf = 0.15,
                                     iou = 0.30,
                                     imgsz = 1280)[0]
        
        ball_dict = {}
        boxes = results.boxes

        if boxes is not None and len(boxes):
            h, w = frame.shape[:2]
            cands = []

            def center(b):
                x1,y1,x2,y2 = b
                return (0.5*(x1+x2), 0.5*(y1+y2))

            for b in boxes:
                xyxy = b.xyxy[0].cpu().numpy()
                x1,y1,x2,y2 = xyxy
                conf = float(b.conf[0])
                bw, bh = x2 - x1, y2 - y1
                area = bw * bh
                ar = bw / max(bh, 1e-6)

                # gate: small & roughly square
                if area <= 0.015 * w * h and 0.7 <= ar <= 1.5:
                    cands.append((conf, area, xyxy))

            if cands:
                if getattr(self, "prev_center", None) is not None:
                    pcx, pcy = self.prev_center
                    # nearest to last center, then higher conf, then smaller area
                    cands.sort(key=lambda t: ((center(t[2])[0]-pcx)**2 + (center(t[2])[1]-pcy)**2, -t[0], t[1]))
                else:
                    # first frame: highest conf, then smallest
                    cands.sort(key=lambda t: (-t[0], t[1]))

                chosen = cands[0][2]
                ball_dict[1] = chosen.tolist()
                self.prev_center = center(chosen)
        else:
            # optional: keep last box for continuity when nothing is detected
            pass

        return ball_dict
       
        # --- End of potential fix ---

    
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
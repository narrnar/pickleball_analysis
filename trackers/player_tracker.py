from ultralytics import YOLO
import cv2
import pickle
import sys
sys.path.append('../')
from utils import measure_distance, get_center_of_bbox

class PlayerTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def choose_and_filter_players(self, court_keypoints, player_detections):
        player_deterctions_first_name = player_detections[0]
        chosen_player = self.choose_players(court_keypoints, player_deterctions_first_name)
        filtered_player_detections = []
        for player_dict in player_detections:
            filtered_player_dict = {track_id: bbox for track_id, bbox in player_dict.items() if track_id in chosen_player}
            filtered_player_detections.append(filtered_player_dict)
        return filtered_player_detections


    def choose_players(self, court_keypoints, player_dict):
        # Logic to choose players based on court keypoints and first frame detections
        distances = []
        for track_id, bbox in player_dict.items():
            player_cetner = get_center_of_bbox(bbox)

            # Calculate distance from player to a reference court keypoint (e.g., center of the court)
            min_distance = float('inf')
            for i in range(0, len(court_keypoints), 2):
                court_keypoint = (court_keypoints[i], court_keypoints[i+1])
                distance = measure_distance(player_cetner, court_keypoint)
                if distance < min_distance:
                    min_distance = distance
            distances.append((track_id, min_distance))

        # Sort players by distance in ascending order
        distances.sort(key=lambda x: x[1])
        # Choose the first four players/tracks
        # chosen_players = [distances[0][0], distances[1][0], distances[2][0], distances[3][0]]
        # Choose the first two players/tracks
        chosen_players = [distances[0][0], distances[1][0]]
        return chosen_players
            


    def detect_frames(self, frames, read_from_stub = False, stub_path = None):
        player_detections = []

        # If reading from stub, load detections from pickle file
        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                player_detections = pickle.load(f)
            return player_detections

        for frame in frames:
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict)

        # Save detections to stub file if path is provided
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(player_detections, f)

        return player_detections


    def detect_frame(self, frame):
        # results = self.model.track(frame, persist = True)[0]
        # id_name_dict = results.names

        # player_dict = {}
        # for box in results.boxes:
        #     track_id = int(box.id.tolist()[0])
        #     result = box.xyxy.tolist()[0]
        #     object_cls_id = box.cls.tolist()[0]
        #     object_cls_name = id_name_dict[object_cls_id]
        #     if object_cls_name == "person":
        #         player_dict[track_id] = result

        # return player_dict

        # --- Potential fix for tracking player issues ---
        results = self.model.track(
            frame,
            persist=True,
            classes=[0], # person only
            conf=0.45, 
            iou=0.5,
            imgsz=960,     
            vid_stride=1, # don't skip frames
            tracker='bytetrack.yaml'
        )

        res = results[0]
        boxes = res.boxes
        id_name = results[0].names 

        player_dict = {}
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            ids  = (boxes.id.cpu().numpy() if boxes.id is not None else None)
            cls  = (boxes.cls.cpu().numpy() if boxes.cls is not None else None)

            # Keep only "person" (defensive check) and take the two tallest (closest)
            keep = []
            for j in range(xyxy.shape[0]):
                if cls is None or int(cls[j]) == 0:
                    x1,y1,x2,y2 = xyxy[j]
                    h = y2 - y1
                    keep.append((h, j))
            keep.sort(reverse=True)
            keep = keep[:2]

            for _, j in keep:
                tid = int(ids[j]) if ids is not None else j  # fallback to index if no ID
                player_dict[tid] = xyxy[j].tolist()

        return player_dict
        # --- End of potential fix ---

    

    def draw_bboxes(self, video_frames, player_detections):
        output_video_frames = []
        for frame, player_dict in zip(video_frames, player_detections):
            # Draw player bounding boxes
            for track_id, bbox in player_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f"Player ID: {track_id}", (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            output_video_frames.append(frame)

        return output_video_frames
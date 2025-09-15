# Import necessary modules
from utils import (read_video, 
                   save_video)
from trackers import PlayerTracker, BallTracker

def main():
    # Read Video
    input_video_path = 'input_videos/input_video.mp4'
    video_frames = read_video(input_video_path)


    # Detect Players and Ball
    player_tracker = PlayerTracker(model_path='yolov8x.pt')
    ball_tracker = BallTracker(model_path='models/yolov8n_last.pt')
    player_detections = player_tracker.detect_frames(video_frames,
                                                     read_from_stub=True, # Run False for first time or to re-generate stubs
                                                     stub_path='tracker_stubs/player_detections.pkl'
                                                     )
    ball_detections = ball_tracker.detect_frames(video_frames,
                                                     read_from_stub=True, # Run False for first time or to re-generate stubs
                                                     stub_path='tracker_stubs/ball_detections.pkl'
                                                     )

    # Court Line Detector Model

    # Choose Players

    # MiniCourt

    # Detect Ball Shots

    # Convert positions to mini court positions


    # --- Draw Output ---

    # -- Draw Player Bounding Boxes
    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detections)
    output_video_frames = ball_tracker.draw_bboxes(video_frames, ball_detections)


    # -- Draw Court Keypoints

    # -- Draw Mini Court

    # -- Draw Player Stats

    # -- Draw Frame Number on Top Left Corner


    save_video(output_video_frames, 'output_videos/output_video.avi')


if __name__ == "__main__":
    main()
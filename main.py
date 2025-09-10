# Import necessary modules
from utils import (read_video, 
                   save_video)
from trackers import PlayerTracker

def main():
    # Read Video
    input_video_path = 'input_videos/input_video.mp4'
    video_frames = read_video(input_video_path)


    # Detect Players and Ball
    player_tracker = PlayerTracker(model_path='yolov8x.pt')
    player_detections = player_tracker.detect_frames(video_frames)

    # Court Line Detector Model

    # Choose Players

    # MiniCourt

    # Detect Ball Shots

    # Convert positions to mini court positions


    # --- Draw Output ---

    # -- Draw Player Bounding Boxes
    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detections)

    # -- Draw Court Keypoints

    # -- Draw Mini Court

    # -- Draw Player Stats

    # -- Draw Frame Number on Top Left Corner


    save_video(output_video_frames, 'output_videos/output_video.avi')


if __name__ == "__main__":
    main()
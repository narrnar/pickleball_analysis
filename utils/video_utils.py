import cv2

def read_video(video_path):
    # Reads a video from the specified path and returns a list of frames.
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def save_video(output_video_frames, output_video_path):
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))

    # catch silent failures
    if not out.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for {output_video_path}")
    
    for frame in output_video_frames:
        out.write(frame)
    out.release()
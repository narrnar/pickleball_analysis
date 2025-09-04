# For YOLO Inference
# import pathlib
# print("FILE:", pathlib.Path(__file__).resolve()) # Resolve file path if there are any issues

from ultralytics import YOLO

model = YOLO('models/yolov8n_last.pt')

result = model.predict('input_videos/input_video.mp4', conf = 0.2, save = True)

# Output
print(result)
print("boxes:")
for box in result[0].boxes:
    print(box)
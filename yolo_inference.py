# For YOLO Inference
from ultralytics import YOLO

model = YOLO('yolov8x')

result = model.track('INPUTVIDEO.mp4', conf = 0.2, save = True)

# Output
# print(result)
# print("boxes:")
# for box in result[0].boxes:
#     print(box)
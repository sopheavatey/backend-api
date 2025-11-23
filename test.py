from helper.ocr import run_prediction

YOLO_MODEL_PATH = "./model/YOLO.pt"
CRNN_CHECKPOINT_PATH = "./model/CRNN.pth"
local_image_path = "test.jpg"


text = run_prediction(
    YOLO_MODEL_PATH, 
    CRNN_CHECKPOINT_PATH, 
    local_image_path
)
with open("output.txt", "w", encoding="utf-8") as f:
    f.write(text)
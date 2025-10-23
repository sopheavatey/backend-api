import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
from ultralytics import YOLO
import numpy as np

# =====================================================================================
# STEP 1: CONFIGURATION & CHARACTER SET
# This should match your training script exactly.
# =====================================================================================
class Config:
    IMG_HEIGHT = 40
    IMG_WIDTH = 64

CHAR_LIST = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'ក', 'ខ', 'គ', 'ឃ', 'ង', 'ច', 'ឆ', 'ជ', 'ឈ', 'ញ',
    'ដ', 'ឋ', 'ឌ', 'ឍ', 'ណ', 'ត', 'ថ', 'ទ', 'ធ', 'ន', 'ប', 'ផ', 'ព', 'ភ', 'ម', 'យ', 'រ', 'ល', 'វ',
    'ស', 'ហ', 'ឡ', 'អ', 'ឥ', 'ឦ', 'ឧ', 'ឩ', 'ឪ', 'ឫ', 'ឬ', 'ឭ', 'ឮ', 'ឯ', 'ឰ', 'ឱ', 'ឲ', 'ឳ',
    'ា', 'ិ', 'ី', 'ឹ', 'ឺ', 'ុ', 'ូ', 'ួ', 'ើ', 'ឿ', 'ៀ', 'េ', 'ែ', 'ៃ', 'ោ', 'ៅ', 'ំ', 'ះ', 'ៈ', '៉',
    '៊', '់', '៌', '៍', '៎', '៏', '័', '៑', '្', '។', '៕', '៖', 'ៗ', '៘', '៙', '៚', '៛',
    '០', '១', '២', '៣', '៤', '៥', '៦', '៧', '៨', '៩'
]
char_to_int = {char: i + 1 for i, char in enumerate(CHAR_LIST)}
int_to_char = {i + 1: char for i, char in enumerate(CHAR_LIST)}
NUM_CLASSES = len(char_to_int) + 1

# =====================================================================================
# STEP 2: HELPER CLASSES AND FUNCTIONS
# We need all the same tools we used for training.
# =====================================================================================

# --- Image Transformation ---
class ResizeAndPad:
    def __init__(self, height, width, fill=(0, 0, 0)):
        self.height = height
        self.width = width
        self.fill = fill

    def __call__(self, img):
        original_width, original_height = img.size
        target_aspect = self.width / self.height
        original_aspect = original_width / original_height
        if original_aspect > target_aspect:
            new_width = self.width
            new_height = int(new_width / original_aspect)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            new_img = Image.new(img.mode, (self.width, self.height), self.fill)
            paste_y = (self.height - new_height) // 2
            new_img.paste(img, (0, paste_y))
        else:
            new_height = self.height
            new_width = int(new_height * original_aspect)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            new_img = Image.new(img.mode, (self.width, self.height), self.fill)
            paste_x = (self.width - new_width) // 2
            new_img.paste(img, (paste_x, 0))
        return new_img

# --- Bounding Box Sorting ---
def sort_words_into_lines(boxes):
    if not boxes: return []
    boxes.sort(key=lambda box: box[1])
    lines, current_line = [], [boxes[0]]
    for box in boxes[1:]:
        prev_box = current_line[-1]
        _, prev_y_min, _, prev_y_max = prev_box
        _, current_y_min, _, current_y_max = box
        prev_h = prev_y_max - prev_y_min
        if abs(current_y_min - prev_y_min) < prev_h * 0.7:
            current_line.append(box)
        else:
            current_line.sort(key=lambda b: b[0])
            lines.append(current_line)
            current_line = [box]
    current_line.sort(key=lambda b: b[0])
    lines.append(current_line)
    return lines

# --- CRNN Model Definition (Copied from train.py) ---
class CRNN(nn.Module):
    def __init__(self, num_classes, input_height=40, rnn_hidden_size=256, rnn_layers=1, dropout=0.5):
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            nn.Conv2d(512, 512, kernel_size=(2,2), stride=(1,1), padding=0), nn.BatchNorm2d(512), nn.ReLU(inplace=True)
        )
        self.rnn = nn.LSTM(512, rnn_hidden_size, rnn_layers, bidirectional=True, dropout=dropout if rnn_layers > 1 else 0, batch_first=False)
        self.classifier = nn.Linear(rnn_hidden_size * 2, num_classes)

    def forward(self, x):
        features = self.cnn(x)
        features = features.squeeze(2).permute(2, 0, 1)
        rnn_output, _ = self.rnn(features)
        return self.classifier(rnn_output)

# --- CTC Decoder (Copied from train.py) ---
def decode_predictions(preds, int_to_char_map):
    preds = preds.argmax(dim=2).permute(1, 0)
    decoded_texts = []
    for pred in preds:
        collapsed = [p for i, p in enumerate(pred) if p != 0 and (i == 0 or p != pred[i-1])]
        decoded_texts.append("".join([int_to_char_map.get(c.item(), '') for c in collapsed]))
    return decoded_texts

# =====================================================================================
# STEP 3: THE MAIN PREDICTION PIPELINE (PUBLIC)
# =====================================================================================
def run_prediction(yolo_model_path: str, crnn_checkpoint_path: str, source_image_path: str) -> str:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. Load Models ---
    print("Loading models...")
    yolo_model = YOLO(yolo_model_path)
    
    crnn_model = CRNN(num_classes=NUM_CLASSES, input_height=Config.IMG_HEIGHT).to(device)
    checkpoint = torch.load(crnn_checkpoint_path, map_location=device)
    crnn_model.load_state_dict(checkpoint['model_state_dict'])
    crnn_model.eval() # Set model to evaluation mode
    print("Models loaded successfully.")

    # --- 2. Define Image Transformation ---
    transform = transforms.Compose([
        ResizeAndPad(Config.IMG_HEIGHT, Config.IMG_WIDTH),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # --- 3. Detect and Sort Words ---
    print("Running word detection...")
    yolo_results = yolo_model.predict(source=source_image_path, save=False, show=False, save_crop=False, conf=0.33, iou=0.7, imgsz=1024)
    boxes_xyxy = yolo_results[0].boxes.xyxy.cpu().numpy().tolist()
    
    if not boxes_xyxy:
        print("No words detected.")
        return

    print(f"Detected {len(boxes_xyxy)} words. Sorting into lines...")
    ordered_lines_of_boxes = sort_words_into_lines(boxes_xyxy)

    # --- 4. Crop, Process, and Batch Words ---
    print("Processing detected words...")
    batch_tensors = []
    main_image = Image.open(source_image_path).convert("RGB")

    for line in ordered_lines_of_boxes:
        for box in line:
            xmin, ymin, xmax, ymax = map(int, box)
            
            # Crop the word from the main image
            cropped_word_img = main_image.crop((xmin, ymin, xmax, ymax))
            
            # Apply the same transformations as training
            transformed_tensor = transform(cropped_word_img)
            batch_tensors.append(transformed_tensor)
            
    # Stack all word tensors into a single batch
    if not batch_tensors:
        print("No valid words to process after cropping.")
        return
        
    batch = torch.stack(batch_tensors, 0).to(device)
    print(f"Created a batch of {len(batch)} word images.")

    # --- 5. Run CRNN Recognition ---
    print("Running text recognition...")
    with torch.no_grad():
        preds = crnn_model(batch)
    
    # --- 6. Decode and Reconstruct Text ---
    decoded_word_list = decode_predictions(preds, int_to_char)
    
    print("\n--- Recognition Complete ---")
    final_text = []
    word_counter = 0
    for line in ordered_lines_of_boxes:
        line_text = "".join(decoded_word_list[word_counter : word_counter + len(line)])
        final_text.append(line_text)
        word_counter += len(line)

    # Print the final result
    full_page_text = "\n".join(final_text)
    print("Recognized Text:\n")
    print(full_page_text)
    print("\n--------------------------\n")

    return full_page_text


# =====================================================================================
# EXAMPLE USAGE
# =====================================================================================
# if __name__ == '__main__':
#     # --- IMPORTANT: UPDATE THESE PATHS ---
#     YOLO_MODEL_PATH = "./yolo_best.pt" # Your trained YOLOv8 model
#     CRNN_CHECKPOINT_PATH = "./crnn_epoch_2.pth" # Your trained CRNN checkpoint
#     TEST_IMAGE_PATH = "./test.png" # The image you want to recognize

#     if not os.path.exists(YOLO_MODEL_PATH):
#         print(f"Error: YOLO model not found at '{YOLO_MODEL_PATH}'")
#     elif not os.path.exists(CRNN_CHECKPOINT_PATH):
#         print(f"Error: CRNN checkpoint not found at '{CRNN_CHECKPOINT_PATH}'")
#     elif not os.path.exists(TEST_IMAGE_PATH):
#         print(f"Error: Test image not found at '{TEST_IMAGE_PATH}'")
#     else:
#         text = run_prediction(YOLO_MODEL_PATH, CRNN_CHECKPOINT_PATH, TEST_IMAGE_PATH)
        
#         with open("output.txt", "w", encoding="utf-8") as f:
#             f.write(text)

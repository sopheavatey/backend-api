import PIL.Image as Image
from torchvision import transforms

class Config: # OUR MODEL ONLY SUPPORT THIS SIZE
    IMG_HEIGHT = 40
    IMG_WIDTH = 64

def sort_words_into_lines(boxes):
    """
    Sorts bounding boxes into lines based on their center coordinates.
    
    Args:
        boxes (list): List of [x1, y1, x2, y2] coordinates.
        
    Returns:
        list: List of lines, where each line is a list of sorted boxes.
    """
    if not boxes:
        return []

    # Helper to get center y and height
    def get_center_y(box):
        return (box[1] + box[3]) / 2

    def get_center_x(box):
        return (box[0] + box[2]) / 2

    def get_height(box):
        return box[3] - box[1]

    # 1. Initial Sort: Sort all boxes by their Y-center (Top-to-Bottom)
    boxes.sort(key=lambda b: get_center_y(b))

    lines = []
    current_line = [boxes[0]]

    # 2. Group into lines
    for box in boxes[1:]:
        prev_box = current_line[-1]
        
        # Calculate centers and height
        cy = get_center_y(box)
        prev_cy = get_center_y(prev_box)
        prev_h = get_height(prev_box)

        # Threshold: If the vertical distance between centers is less than 
        # half the height of the previous character, consider it the same line.
        # You can adjust 0.5 (50%) if our text is very wavy or tight.
        if abs(cy - prev_cy) < prev_h * 0.5:
            current_line.append(box)
        else:
            # Finish the current line
            # Sort the current line by X-center (Left-to-Right)
            current_line.sort(key=lambda b: get_center_x(b))
            lines.append(current_line)
            
            # Start a new line
            current_line = [box]

    # 3. Append and sort the final line
    current_line.sort(key=lambda b: get_center_x(b))
    lines.append(current_line)

    return lines


# =====================================================================================
# RESIZE AND PADDING THE IMAGE PREPARE FOR CRNN
# =====================================================================================
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


# =====================================================================================
# THE MAIN PREDICTION PIPELINE
# =====================================================================================
def run_prediction(yolo_model_path, crnn_checkpoint_path, source_image_path, mode='production'):
    if mode == "production":
        from helper.yolo_inference import YOLOPredictor
        from helper.crnn_inference import CRNNPredictor
    else:
        from helper.yolo_inference_old import YOLOPredictor
        from helper.crnn_inference_old import CRNNPredictor

    # =====================================================================================
    # STEP 1: LOAD THE MODEL
    # =====================================================================================
    print("Loading models...")
    yolo_model = YOLOPredictor(yolo_model_path)
    crnn_model = CRNNPredictor(crnn_checkpoint_path)
    print("Models loaded successfully.")

    # =====================================================================================
    # STEP 2: RUNNING YOLO DETECTION
    # =====================================================================================
    boxes_xyxy = yolo_model.predict(
        image_path=source_image_path,
        conf=0.25,
        iou=0.7
    )
    
    if not boxes_xyxy:
        print("No words detected.")
        return ""

    # =====================================================================================
    # STEP 3: SORTING THE DETECTION OUTPUT
    # =====================================================================================
    ordered_lines_of_boxes = sort_words_into_lines(boxes_xyxy)

    # =====================================================================================
    # STEP 4: RESIZE-PADDING-NORMALIZE THEN CONVERT IT TO BATCH OF TENSORS
    # =====================================================================================
    transform = transforms.Compose([
            ResizeAndPad(Config.IMG_HEIGHT, Config.IMG_WIDTH),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    batch_tensors = []
    main_image = Image.open(source_image_path).convert("RGB")

    for line in ordered_lines_of_boxes:
        for box in line:
            xmin, ymin, xmax, ymax = map(int, box)
            cropped_word_img = main_image.crop((xmin, ymin, xmax, ymax))
            transformed_tensor = transform(cropped_word_img)
            batch_tensors.append(transformed_tensor)

    # =====================================================================================
    # STEP 5: RUNNING CRNN RECOGNITION
    # =====================================================================================
    decoded_word_list = crnn_model.predict(batch_tensor=batch_tensors)
    
    # =====================================================================================
    # STEP 6: RECONSTRUCT THE LINE OF TEXT
    # =====================================================================================
    final_text = []
    word_counter = 0
    for line in ordered_lines_of_boxes:
        line_text = "".join(decoded_word_list[word_counter : word_counter + len(line)])
        final_text.append(line_text)
        word_counter += len(line)

    # OPTIONAL PRINT OUTPUT
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
#     YOLO_MODEL_PATH = "yolo_best_inference.pt" # Our trained YOLOv8 model
#     CRNN_CHECKPOINT_PATH = "crnn_best.onnx" # Our trained CRNN checkpoint
#     TEST_IMAGE_PATH = "./test1.jpg" # The image you want to recognize

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

import torch

from yolo_inference import YOLOPredictor
from crnn_inference import CRNNPredictor

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
        # You can adjust 0.5 (50%) if your text is very wavy or tight.
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
# STEP 3: THE MAIN PREDICTION PIPELINE
# =====================================================================================
def run_prediction(yolo_model_path, crnn_checkpoint_path, source_image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading models...")
    yolo_model = YOLOPredictor(yolo_model_path)
    crnn_model = CRNNPredictor(crnn_checkpoint_path)
    print("Models loaded successfully.")

    # print("Running word detection...")
    boxes_xyxy = yolo_model.predict(
        image_path=source_image_path,
        conf=0.25,
        iou=0.7
    )
    
    if not boxes_xyxy:
        print("No words detected.")
        return ""

    ordered_lines_of_boxes = sort_words_into_lines(boxes_xyxy)

    decoded_word_list = crnn_model.predict(
        image_path=source_image_path, 
        lines_of_boxes=ordered_lines_of_boxes
    )
    
    # print("\n--- Recognition Complete ---")
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
#     YOLO_MODEL_PATH = "yolo_best_inference.pt" # Your trained YOLOv8 model
#     CRNN_CHECKPOINT_PATH = "crnn_best_inference.pth" # Your trained CRNN checkpoint
#     TEST_IMAGE_PATH = "./test2.jpg" # The image you want to recognize

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

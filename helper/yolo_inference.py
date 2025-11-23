import torch
import cv2
import numpy as np
import onnxruntime as ort

# --- IMPORTS FROM YOUR FILES ---
from helper.util import non_max_suppression
from helper.dataset import resize

# ==========================================
# 1. PREDICTOR CLASS
# ==========================================
class YOLOPredictor:
    def __init__(self, onnx_model_path, providers=['CPUExecutionProvider']):
        """
        Initializes the ONNX Runtime session for YOLO.
        """
        print(f"Loading ONNX YOLO from {onnx_model_path}...")
        self.session = ort.InferenceSession(onnx_model_path, providers=providers)
        
        # Get input/output names dynamically
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        print("ONNX Model loaded.")

    def predict(self, image_path, conf=0.25, iou=0.7, img_size=640):
        # --- A. Preprocessing (Using dataset.py) ---
        img_cv = cv2.imread(image_path)
        if img_cv is None: return []
        
        # Use your dataset.py resize function
        # It returns: image, (ratio_x, ratio_y), (pad_w, pad_h)
        img_resized, ratio, pad = resize(img_cv, img_size, augment=False)
        
        # Prepare Blob for ONNX (HWC -> CHW, BGR -> RGB, Normalize)
        # ONNX expects a Float32 NumPy array, not a Tensor
        blob = img_resized.transpose((2, 0, 1))[::-1]  # HWC -> CHW, BGR -> RGB
        blob = np.ascontiguousarray(blob).astype(np.float32)
        blob /= 255.0
        blob = np.expand_dims(blob, axis=0)  # Add batch dimension [1, 3, 640, 640]

        # --- B. Inference (ONNX) ---
        # This replaces 'preds = self.model(img_tensor)'
        outputs = self.session.run([self.output_name], {self.input_name: blob})
        
        # ONNX Output is a NumPy array [1, 5, 8400]
        raw_preds_numpy = outputs[0]

        # --- C. Post-Processing ---
        # We convert back to PyTorch tensor ONLY for NMS because
        # your 'non_max_suppression' function relies on torchvision.ops.nms
        preds_tensor = torch.from_numpy(raw_preds_numpy)
        # --- D. Post-Processing (Using util.py) ---
        # non_max_suppression handles score filtering and NMS
        nms_output = non_max_suppression(preds_tensor, conf, iou)
        
        det = nms_output[0] # Get first batch item
        if det is None or len(det) == 0:
            return []
            
        # --- E. Rescaling (The only manual math needed) ---
        # We reverse the 'resize' operations: subtract pad, then divide by ratio
        boxes = det[:, :4].clone()
        pad_w, pad_h = pad
        ratio_w, ratio_h = ratio
        
        boxes[:, 0] -= pad_w  # x1
        boxes[:, 1] -= pad_h  # y1
        boxes[:, 2] -= pad_w  # x2
        boxes[:, 3] -= pad_h  # y2
        
        boxes[:, [0, 2]] /= ratio_w
        boxes[:, [1, 3]] /= ratio_h
        
        # Clip to original image dimensions
        h_orig, w_orig = img_cv.shape[:2]
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, w_orig)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, h_orig)
        
        return boxes.cpu().numpy().tolist()

import torch
import cv2
import numpy as np

from model_def.MyYolo import MyYolo
from helper.util import non_max_suppression
from helper.dataset import resize

# ==========================================
# 1. PREDICTOR CLASS
# ==========================================
class YOLOPredictor:
    def __init__(self, weights_path, device='cuda'):
        self.device = device
        print(f"Loading Custom YOLO from {weights_path}...")
        
        self.model = MyYolo(num_class=1)
        self.model.to(self.device)
        
        # Load Weights
        checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)
        if 'model' in checkpoint:
            state_dict = checkpoint['model'].state_dict() if hasattr(checkpoint['model'], 'state_dict') else checkpoint['model']
        else:
            state_dict = checkpoint
            
        self.model.load_state_dict(state_dict, strict=False)
        self.model.build(input_size=640)
        self.model.eval()
        print("Model loaded.")

    def predict(self, image_path, conf=0.25, iou=0.7, img_size=640):
        # --- A. Preprocessing (Using dataset.py) ---
        img_cv = cv2.imread(image_path)
        if img_cv is None: return []
        
        # Use your dataset.py resize function
        # It returns: image, (ratio_x, ratio_y), (pad_w, pad_h)
        img_resized, ratio, pad = resize(img_cv, img_size, augment=False)
        
        # Convert to Tensor (Logic from Dataset.__getitem__)
        img_tensor = img_resized.transpose((2, 0, 1))[::-1]  # HWC->CHW, BGR->RGB
        img_tensor = np.ascontiguousarray(img_tensor)
        img_tensor = torch.from_numpy(img_tensor).to(self.device).float()
        img_tensor /= 255.0
        img_tensor = img_tensor.unsqueeze(0)

        # --- B. Inference ---
        with torch.no_grad():
            preds = self.model(img_tensor)

        # --- C. Post-Processing (Using util.py) ---
        # non_max_suppression handles score filtering and NMS
        nms_output = non_max_suppression(preds, conf, iou)
        
        det = nms_output[0] # Get first batch item
        if det is None or len(det) == 0:
            return []
            
        # --- D. Rescaling (The only manual math needed) ---
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

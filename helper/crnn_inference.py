import torch
from torchvision import transforms
from PIL import Image
from model_def.MyCRNN import CRNN

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
    '០', '១', '២', '៣', '៤', '៥', '៦', '៧', '៨', '៩', '[OOV]'
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

class CRNNPredictor:
    def __init__(self, weights_path, device='cuda'):
        self.device = device
        print(f"Loading Custom CRNN from {weights_path}...")
        
        self.model = CRNN(num_classes=NUM_CLASSES, input_height=Config.IMG_HEIGHT)
        self.model.to(self.device)
        
        # Load Weights
        # Load Weights
        checkpoint = torch.load(weights_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
            
        self.model.load_state_dict(state_dict)
        self.model.eval()
        print("Model loaded.")

    # --- CTC Decoder (Copied from train.py) ---
    def decode_predictions(self, preds, int_to_char_map):
        preds = preds.argmax(dim=2).permute(1, 0)
        decoded_texts = []
        for pred in preds:
            collapsed = [p for i, p in enumerate(pred) if p != 0 and (i == 0 or p != pred[i-1])]
            decoded_texts.append("".join([int_to_char_map.get(c.item(), '') for c in collapsed]))
        return decoded_texts

    def predict(self, image_path, lines_of_boxes):
        transform = transforms.Compose([
            ResizeAndPad(Config.IMG_HEIGHT, Config.IMG_WIDTH),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        batch_tensors = []
        main_image = Image.open(image_path).convert("RGB")

        for line in lines_of_boxes:
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
            return []
            
        batch = torch.stack(batch_tensors, 0).to(self.device)
        print(f"Created a batch of {len(batch)} word images.")

        # --- 5. Run CRNN Recognition ---
        print("Running text recognition...")
        with torch.no_grad():
            preds = self.model(batch)
        
        # --- 6. Decode and Reconstruct Text ---
        return self.decode_predictions(preds, int_to_char)
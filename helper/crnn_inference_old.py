import torch
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

class CRNNPredictor:
    def __init__(self, weights_path, device='cuda'):
        self.device = device
        print(f"Loading Custom CRNN from {weights_path}...")
        
        self.model = CRNN(num_classes=NUM_CLASSES, input_height=Config.IMG_HEIGHT)
        self.model.to(self.device)
        
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

    def predict(self, batch_tensor):
        with torch.no_grad():
            preds = self.model(batch_tensor)
        
        return self.decode_predictions(preds, int_to_char)
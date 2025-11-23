import onnxruntime as ort
import numpy as np
from torch import Tensor

# =====================================================================================
# CONFIGURATION & CHARACTER SET
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
int_to_char = {i + 1: char for i, char in enumerate(CHAR_LIST)}

class CRNNPredictor:
    def __init__(self, onnx_model_path, providers=['CPUExecutionProvider']):
        """
        Initializes the ONNX Runtime session.
        Args:
            onnx_model_path (str): Path to the .onnx file.
            providers (list): List of execution providers (default CPU).
        """
        print(f"Loading ONNX CRNN from {onnx_model_path}...")
        self.session = ort.InferenceSession(onnx_model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        print("ONNX Model loaded.")

    def decode_predictions(self, preds):
        """
        Decodes the CTC output from the model.
        Args:
            preds (np.ndarray): Output from ONNX model [Seq, Batch, Class]
        Returns:
            list[str]: List of decoded strings.
        """
        # ONNX output is [Seq, Batch, Class].
        # We need argmax over classes -> [Seq, Batch]
        preds_idx = np.argmax(preds, axis=2)
        
        # Transpose to [Batch, Seq] for easy iteration
        preds_idx = preds_idx.transpose(1, 0)
        
        decoded_texts = []
        for seq in preds_idx:
            decoded_seq = []
            prev_char = 0
            for char_idx in seq:
                if char_idx != 0 and char_idx != prev_char:
                    decoded_seq.append(int_to_char.get(char_idx, ''))
                prev_char = char_idx
            decoded_texts.append("".join(decoded_seq))
        return decoded_texts

    def predict(self, batch_tensor):
        """
        Runs inference on a batch of images.
        Args:
            batch_tensor (torch.Tensor): Input tensor of shape [B, 3, H, W].
        Returns:
            list[str]: List of recognized text strings.
        """
        # 1. Convert PyTorch Tensor to Numpy
        # We assume the tensor is already normalized and on CPU (or we move it)
        if isinstance(batch_tensor, Tensor):
            numpy_input = batch_tensor.detach().cpu().numpy()
        else:
            numpy_input = batch_tensor

        # 2. Run ONNX Inference
        # Input shape must be [B, 3, 40, 64]
        outputs = self.session.run(None, {self.input_name: numpy_input})
        
        # Output[0] is the logits [Seq, Batch, Class]
        logits = outputs[0]

        # 3. Decode
        return self.decode_predictions(logits)
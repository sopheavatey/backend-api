import torch.nn as nn

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

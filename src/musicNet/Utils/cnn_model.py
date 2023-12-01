import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNMusicNet(nn.Module):
    def __init__(self):
        super(CNNMusicNet, self).__init__()

        # Convolutional layers
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=(3,3), stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(8, 32, kernel_size=(3,3), stride=1, padding=1),
            nn.LeakyReLU(),
            torch.nn.MaxPool2d(2),
            nn.Dropout(0.2),
            nn.Conv2d(32, 32, kernel_size=(3,3), stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2)
        )

        self.avg_pooling = nn.AdaptiveAvgPool2d((7, 7))

        # Linear layers
        self.classifier = nn.Sequential(
            nn.Linear(3136, 256),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10)
        )
        
    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.avg_pooling(x)
        f = nn.Flatten()
        x = f(x)
        x = self.classifier(x)
        s = nn.Softmax()
        x = s(x)
        return x

model = CNNMusicNet()

from torchsummary import summary
print(summary(model, (3, 128, 128)))
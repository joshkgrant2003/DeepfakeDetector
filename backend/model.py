import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

class FrequencyCNN(nn.Module):
    def __init__(self, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(128, out_dim)

    def forward(self, x):
        x = self.net(x)
        x = x.flatten(1)
        return self.fc(x)
    
class DualBranchDeepfakeDetector(nn.Module):
    def __init__(self):
        super().__init__()

        backbone = models.efficientnet_b0(weights=None)
        self.spatial = backbone.features
        self.spatial_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.spatial_dim = 1280

        self.freq_branch = FrequencyCNN(out_dim=256)

        self.spatial_weight = 0.8
        self.freq_weight = 0.2

        fusion_dim = self.spatial_dim + 256
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def fft_transform(self, x):
        fft = torch.fft.fft2(x, dim=(-2, -1))
        fft = torch.fft.fftshift(fft, dim=(-2, -1))
        return torch.log1p(torch.abs(fft))

    def forward(self, x):
        s = self.spatial(x)
        s = self.spatial_pool(s).flatten(1)

        fx = self.fft_transform(x)
        f = self.freq_branch(fx)

        s = self.spatial_weight * s
        f = self.freq_weight * f

        z = torch.cat([s, f], dim=1)
        return self.classifier(z).squeeze(1)

def load_model():
    checkpoints = torch.load("weights.pth", map_location=torch.device('cpu'), weights_only=False)
    model = DualBranchDeepfakeDetector()
    model.load_state_dict(checkpoints["model_state_dict"])
    model.eval()
    return model

transform = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def predict_image(model, image: Image.Image):
    x = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(x)
        prob = torch.sigmoid(output).item()
    return {
        "label": "Real" if prob < 0.5 else "Fake",
        "confidence": float(prob)
    }
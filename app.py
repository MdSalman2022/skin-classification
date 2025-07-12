import torch
import torchvision.models as models
import torch.nn.functional as F
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image
import gradio as gr
import os

IMAGE_SIZE = 224
NUM_CLASSES = 7
DEVICE = 'cpu'
class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

class TinyViT5M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.model.head = torch.nn.Identity()
        self.feature_dim = 768
    def forward(self, x):
        return self.model(x)

class AntiOverfittingEnsemble(torch.nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.efficientnet = models.efficientnet_b0(weights=None)
        self.efficientnet.classifier[1] = torch.nn.Linear(self.efficientnet.classifier[1].in_features, num_classes)
        self.mobilenet = models.mobilenet_v3_large(weights=None)
        self.mobilenet.classifier[3] = torch.nn.Linear(self.mobilenet.classifier[3].in_features, num_classes)
        self.tinyvit = TinyViT5M()
        fusion_dim = 384
        self.fusion = torch.nn.Sequential(
            torch.nn.Linear(1280 + 768 + 960, fusion_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(fusion_dim, num_classes)
        )
    def forward(self, x):
        eff_features = self.efficientnet.features(x)
        eff_pooled = F.adaptive_avg_pool2d(eff_features, (1, 1)).flatten(1)
        mob_features = self.mobilenet.features(x)
        mob_pooled = F.adaptive_avg_pool2d(mob_features, (1, 1)).flatten(1)
        vit_features = self.tinyvit(x)
        all_features = torch.cat([eff_pooled, vit_features, mob_pooled], dim=1)
        logits = self.fusion(all_features)
        return logits

model_path = os.path.join(os.path.dirname(__file__), 'dynamic_ensemble_model_complete.pth')
model = AntiOverfittingEnsemble()
model.load_state_dict(torch.load(model_path, map_location=DEVICE))
model.to(DEVICE)
model.eval()

test_transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

def predict(image):
    image = image.convert('RGB')
    image_np = np.array(image)
    transformed = test_transform(image=image_np)
    tensor = transformed['image'].unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)[0]
        pred_idx = torch.argmax(probs).item()
        pred_class = class_names[pred_idx]
        confidence = probs[pred_idx].item() * 100
        prob_dict = {class_names[i]: f"{probs[i].item() * 100:.2f}%" for i in range(NUM_CLASSES)}
    return f"{pred_class.upper()} ({confidence:.2f}%)", prob_dict

gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Label(num_top_classes=7, label="Class Probabilities")
    ],
    title="Dermatology Ensemble Classifier",
    description="Upload a skin image to get a prediction."
).launch()
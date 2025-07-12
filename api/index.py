import io
import torch
import torchvision.transforms as T
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torchvision.models as models
import os
import torch.nn.functional as F
import timm

# Define constants from your notebook
IMAGE_SIZE = 224
NUM_CLASSES = 7
DEVICE = 'cpu'  # Vercel uses CPU
class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']  # From notebook

# Placeholder for your student model class (COPY THE EXACT CLASS FROM YOUR NOTEBOOK HERE)

# --- Simplified AntiOverfittingEnsemble for Inference ---
class TinyViT5M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Use a small ViT as TinyViT substitute for deployment
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

        # Fusion layer
        fusion_dim = 384
        self.fusion = torch.nn.Sequential(
            torch.nn.Linear(1280 + 768 + 960, fusion_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(fusion_dim, num_classes)
        )

    def forward(self, x):
        # Extract features
        eff_features = self.efficientnet.features(x)
        eff_pooled = F.adaptive_avg_pool2d(eff_features, (1, 1)).flatten(1)
        mob_features = self.mobilenet.features(x)
        mob_pooled = F.adaptive_avg_pool2d(mob_features, (1, 1)).flatten(1)
        vit_features = self.tinyvit(x)

        # Concatenate features
        all_features = torch.cat([eff_pooled, vit_features, mob_pooled], dim=1)
        logits = self.fusion(all_features)
        return logits

# Model loading with repo-relative path
model_path = os.path.join(os.path.dirname(__file__), '..', 'dynamic_ensemble_model_complete.pth')
model = AntiOverfittingEnsemble()
model.load_state_dict(torch.load(model_path, map_location=DEVICE))
model.to(DEVICE)
model.eval()  # Set to evaluation mode

# Preprocessing transform (matching notebook's DermatologyAugmentation for test)
test_transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

app = FastAPI()
static_dir = os.path.join(os.path.dirname(__file__), '..', 'static')
templates_dir = os.path.join(os.path.dirname(__file__), '..', 'templates')
app.mount("/static", StaticFiles(directory=static_dir), name="static")
templates = Jinja2Templates(directory=templates_dir)

# Home page with upload form (UI)
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Prediction endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        image_np = np.array(image)

        # Preprocess
        transformed = test_transform(image=image_np)
        tensor = transformed['image'].unsqueeze(0).to(DEVICE)  # Add batch dim

        # Predict
        with torch.no_grad():
            output = model(tensor)
            probs = torch.softmax(output, dim=1)[0]
            pred_idx = torch.argmax(probs).item()
            pred_class = class_names[pred_idx]
            confidence = probs[pred_idx].item() * 100

        return {
            "prediction": pred_class.upper(),
            "confidence": f"{confidence:.2f}%",
            "probabilities": {class_names[i]: f"{probs[i].item() * 100:.2f}%" for i in range(NUM_CLASSES)}
        }
    except Exception as e:
        return {"error": str(e)}
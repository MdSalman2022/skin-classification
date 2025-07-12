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

# Define constants from your notebook
IMAGE_SIZE = 224
NUM_CLASSES = 7
DEVICE = 'cpu'  # Vercel uses CPU
class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']  # From notebook

# Placeholder for your student model class (COPY THE EXACT CLASS FROM YOUR NOTEBOOK HERE)
# Assuming it's an ensemble of EfficientNet-B0 models. Adjust as per your exact code.
class UnifiedEnsemble(torch.nn.Module):
    def __init__(self, num_models=3, num_classes=NUM_CLASSES):
        super(UnifiedEnsemble, self).__init__()
        self.models = torch.nn.ModuleList([
            models.efficientnet_b0(weights=None) for _ in range(num_models)
        ])
        for model in self.models:
            model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
        # Add any fusion layers or custom logic from your notebook

    def forward(self, x):
        outputs = [model(x) for model in self.models]
        return torch.mean(torch.stack(outputs), dim=0)  # Simple average ensemble

# Load the model (adjust path and loading if it's state_dict only)
model = UnifiedEnsemble()  # Initialize your model class
model.load_state_dict(torch.load('dynamic_ensemble_model_complete.pth', map_location=DEVICE))  # If full model: torch.load('model.pth')
model.to(DEVICE)
model.eval()  # Set to evaluation mode

# Preprocessing transform (matching notebook's DermatologyAugmentation for test)
test_transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")  # For CSS/JS if needed
templates = Jinja2Templates(directory="templates")

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
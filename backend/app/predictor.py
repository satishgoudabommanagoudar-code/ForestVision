import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)

model.load_state_dict(
    torch.load("model/forest_model.pth", map_location=device)
)

model.to(device)
model.eval()

classes = ["Forest", "NonForest"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def predict_forest(image_path):

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1)

        confidence, predicted = torch.max(probabilities, 1)

    return {
        "prediction": classes[predicted.item()],
        "confidence": round(confidence.item() * 100, 2)
    }
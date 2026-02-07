import torch
from torchvision import transforms
from PIL import Image
from backend.model import AlzheimerCNN


device = torch.device("cpu")

# IMPORTANT: match the TRAINED model → 4 classes
model = AlzheimerCNN(num_classes=4)
model.load_state_dict(torch.load("alzheimer_model.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# These names must match your folder names (order from training)
CLASSES = [
    "Mild_Demented",
    "Moderate_Demented",
    "Non_Demented",
    "Very_Mild_Demented"
]

def predict_alzheimer(image_path):
    img = Image.open(image_path)
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img)

    prob = torch.softmax(output, dim=1)[0]

    pred_idx = torch.argmax(prob).item()
    predicted_class = CLASSES[pred_idx]
    confidence = float(prob[pred_idx] * 100)

    return {
        "predicted_class": predicted_class,
        "confidence_percentage": round(confidence, 2)
    }

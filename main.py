from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import torch.quantization # Import the quantization library
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io
from collections import OrderedDict

app = Flask(__name__)

# --- Model Loading and Quantization ---

# 1. Load the original model architecture
model = models.densenet121(weights=None)
num_ftrs = model.classifier.in_features
model.classifier = nn.Linear(num_ftrs, 3)  # 3 classes: Normal, Nevus, Melanoma

# 2. Load the trained weights
if torch.cuda.is_available():
  device = 'cuda'
else:
  device = 'cpu'
checkpoint = torch.load('DenseNetModelV5_3.pth', map_location=torch.device(device))
state_dict = checkpoint['model_state_dict']
new_state_dict = OrderedDict((k[6:] if k.startswith('model.') else k, v) for k, v in state_dict.items())
model.load_state_dict(new_state_dict)
model.eval()

# 3. Create a quantized version of the model (NEW STEP)
# This new model will be used for predictions as it is much faster.
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# --- Preprocessing and Class Labels ---

preprocess = transforms.Compose([
    transforms.Resize((450, 450)),
    transforms.ToTensor(),
])

lesion_map = {
    0: 'Normal',
    1: 'Nevus',
    2: 'Melanoma',
}

# --- Prediction Functions ---

def transform_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return preprocess(image).unsqueeze(0)

def get_prediction(tensor):
    with torch.no_grad():
        # Use the faster QUANTIZED model for inference
        output = quantized_model(tensor)
        probs = torch.nn.functional.softmax(output[0], dim=0)
        idx = probs.argmax().item()
        confidence = probs[idx].item()
        return lesion_map.get(idx, 'Unknown'), confidence

# --- Flask Routes ---

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    img_bytes = file.read()
    tensor = transform_image(img_bytes)
    label, confidence = get_prediction(tensor)
    return jsonify({'prediction': label, 'confidence': round(confidence * 100, 2)})

if __name__ == '__main__':
    app.run(debug=True)
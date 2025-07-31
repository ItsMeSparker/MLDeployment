from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io
from collections import OrderedDict

app = Flask(__name__)

# Preprocessing and label map (these are light, safe to keep global)
preprocess = transforms.Compose([
    transforms.Resize((250, 250)),
    transforms.ToTensor(),
])
lesion_map = {0: 'Normal', 1: 'Nevus', 2: 'Melanoma'}

def transform_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return preprocess(image).unsqueeze(0)

def load_model():
    model = models.densenet121(weights=None)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, 3)

    checkpoint = torch.load('DenseNetModelV5_3.pth', map_location='cpu')
    state_dict = checkpoint['model_state_dict']
    new_state_dict = OrderedDict((k[6:] if k.startswith('model.') else k, v) for k, v in state_dict.items())
    model.load_state_dict(new_state_dict)
    model.eval()
    return model

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    try:
        file = request.files['file']
        image_bytes = file.read()
        tensor = transform_image(image_bytes)

        model = load_model()  # 🔁 load on demand
        with torch.no_grad():
            output = model(tensor)
            probs = torch.nn.functional.softmax(output[0], dim=0)
            idx = probs.argmax().item()
            confidence = probs[idx].item()
            return jsonify({
                'prediction': lesion_map.get(idx, 'Unknown'),
                'confidence': round(confidence * 100, 2)
            })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

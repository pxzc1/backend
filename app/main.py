import io
import json
import onnxruntime as ort
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

with open("class_to_idx.json", "r") as f:
    class_to_idx = json.load(f)["class_to_idx"]

idx_to_class = {v: k for k, v in class_to_idx.items()}

session = ort.InferenceSession(
    "model.onnx",
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name

def preprocess(img: Image.Image):
    img = img.convert("RGB")
    img = img.resize((224, 224))

    img = np.array(img).astype("float32") / 255.0
    img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)
    return img

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes))

        input_tensor = preprocess(img)
        outputs = session.run(None, {input_name: input_tensor})[0]

        # Softmax and prediction
        exp = np.exp(outputs)
        probs = exp / np.sum(exp, axis=1, keepdims=True)

        pred_idx = int(np.argmax(probs, axis=1)[0])
        confidence = float(probs[0][pred_idx] * 100)

        return {
            "class": idx_to_class[pred_idx],
            "confidence": round(confidence, 2)
        }

    except Exception as e:
        return {"error": str(e)}
    
@app.get("/")
def home():
    return {"status": "Model API running"}
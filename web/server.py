import base64

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile, Form
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
from torchvision import transforms

from models import create_model
from options.test_options import TestOptions
from util.util import tensor2labelim, tensor2confidencemap, confidencemap2rgboverlay, get_surface_normals

app = FastAPI()

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class MockDataset:
    def __init__(self):
        self.num_labels = 2


opt = TestOptions()
opt.threads = 1
opt.batch_size = 1
opt.serial_batches = True
opt.isTrain = False

mock_dataset = MockDataset()
model = create_model(opt, mock_dataset)
model.setup()
model.eval()


@app.post("/api/predict")
async def predict(
    file1: UploadFile = File(...),
    file2: UploadFile = File(...),
    fx: float = Form(...),
    cx: float = Form(...),
    fy: float = Form(...),
    cy: float = Form(...)
):
    try:
        # Read the uploaded files as numpy arrays
        rgb_orig = cv2.imdecode(np.frombuffer(await file1.read(), np.uint8), cv2.IMREAD_COLOR)
        depth = cv2.imdecode(np.frombuffer(await file2.read(), np.uint8), cv2.IMREAD_ANYDEPTH)
        k = np.array([[float(fx), 0, float(cx)], [0, float(fy), float(cy)], [0, 0, 1]])
        orig_height, orig_width, _ = rgb_orig.shape

        rgb = cv2.resize(rgb_orig, (opt.resize_width, opt.resize_height))
        rgb = rgb.astype(np.float32) / 255

        sne = get_surface_normals(depth, k)
        sne = cv2.resize(sne, (opt.resize_width, opt.resize_height))

        rgb = transforms.ToTensor()(rgb).unsqueeze(dim=0)
        sne = transforms.ToTensor()(sne).unsqueeze(dim=0)

        # Run the prediction
        with torch.no_grad():
            pred = model.net(rgb, sne)
            prob_map = tensor2confidencemap(pred)
            prob_map = cv2.resize(prob_map, (orig_width, orig_height))
            overlay = confidencemap2rgboverlay(rgb_orig, prob_map)

            # Encode the image as base64 strings
            _, img1_bytes = cv2.imencode('.jpg', overlay)
            img1_base64 = base64.b64encode(img1_bytes).decode('utf-8')

            return JSONResponse({
                'img1': img1_base64,
            })
    except Exception as e:
        print(e, flush=True)
        return JSONResponse({
            'error': "An error has occured. Try again later."
        }, status_code=500)

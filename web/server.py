import base64

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
from torchvision import transforms

from models import create_model
from options.test_options import TestOptions
from util.util import tensor2labelim, tensor2confidencemap

app = FastAPI()

origins = ["*"]  # Replace with your own list of allowed origins
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
opt.serial_batches = True  # no shuffle
opt.isTrain = False

mock_dataset = MockDataset()
model = create_model(opt, mock_dataset)
model.setup()
model.eval()


@app.post("/api/predict")
async def predict(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    # Read the uploaded files as numpy arrays
    rgb = cv2.imdecode(np.frombuffer(await file1.read(), np.uint8), cv2.IMREAD_COLOR)
    depth = cv2.imdecode(np.frombuffer(await file2.read(), np.uint8), cv2.IMREAD_ANYDEPTH)

    orig_height, orig_width, _ = rgb.shape

    rgb = cv2.resize(rgb, (opt.resize_width, opt.resize_height))
    rgb = rgb.astype(np.float32) / 255

    depth = depth.astype(np.float32) / 65535
    depth = cv2.resize(depth, (opt.resize_width, opt.resize_height))
    depth = depth[:, :, np.newaxis]

    rgb = transforms.ToTensor()(rgb).unsqueeze(dim=0)
    depth = transforms.ToTensor()(depth).unsqueeze(dim=0)

    # Run the prediction
    with torch.no_grad():
        pred = model.net(rgb, depth)
        palette = 'datasets/palette.txt'
        impalette = list(np.genfromtxt(palette, dtype=np.uint8).reshape(3 * 256))
        pred_img = tensor2labelim(pred, impalette)
        pred_img = cv2.resize(pred_img, (orig_width, orig_height))
        prob_map = tensor2confidencemap(pred)
        prob_map = cv2.resize(prob_map, (orig_width, orig_height))

        # Encode the images as base64 strings
        _, img1_bytes = cv2.imencode('.jpg', pred_img)
        img1_base64 = base64.b64encode(img1_bytes).decode('utf-8')

        _, img2_bytes = cv2.imencode('.jpg', prob_map)
        img2_base64 = base64.b64encode(img2_bytes).decode('utf-8')

        # Return the images as JSON responses
        return JSONResponse({
            'img1': img1_base64,
            'img2': img2_base64,
        })

from typing import Union, Annotated
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import FileResponse
from classifier import detect_large_brown_object
import cv2
import numpy as np
import base64

app = FastAPI()

@app.get("/")
def home():
    return FileResponse("web/index.html");

@app.post("/upload")
async def create_upload_file(file: UploadFile):
    bytes = file.file.read()

    # Decode image bytes to OpenCV image format
    nparr = np.frombuffer(bytes, np.uint8)
    print(nparr)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    is_climbeable, percent = detect_large_brown_object(image, 200)
    return { 'is_climbeable': is_climbeable, 'score': percent }

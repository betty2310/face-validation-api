import base64
from typing import Union
import numpy as np
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, status
import cv2

class ImageRequest(BaseModel):
    """Request model"""
    image_base64: str
    confidence_threshold: float = 0.7
    face_area_threshold: float = 0.1
    
class SuccessResponse(BaseModel):
    """Response model for a successful operation."""
    img: str

class ErrorResponse(BaseModel):
    """Response model for a failed operation."""
    message: str  


PROTOTXT_PATH = "deploy.prototxt"
MODEL_PATH = "res10_300x300_ssd_iter_140000_fp16.caffemodel"


try:
    face_net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, MODEL_PATH)
    if face_net.empty():
        raise FileNotFoundError(f"Could not load Caffe model from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading DNN model: {e}")
    exit()

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post(
    "/face-validation",
    response_model=Union[SuccessResponse, ErrorResponse],
    summary="Validate and crop a face from an image",
    tags=["Face Processing"]
)
def validate_and_crop(request: ImageRequest):
    """
    Accepts a base64 encoded image and performs the following validations:
    1.  Image must contain exactly one face.
    2.  The face area must be > Request.Face_area_ratio of the total image area.

    If successful, it returns a cropped image (double the size of the detected face)
    also encoded in base64.
    """
    # 1. Decode the Base64 string
    try:
        image_bytes = base64.b64decode(request.image_base64)
        # Convert bytes to a numpy array for OpenCV
        np_arr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Could not decode image.")
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"message": "Ảnh sai định dạng"}
        )

    img_height, img_width, _ = image.shape
    img_area = img_width * img_height

    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
    )
    
    face_net.setInput(blob)
    detections = face_net.forward()

    detected_faces = []
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > request.confidence_threshold:
            box = detections[0, 0, i, 3:7] * np.array([img_width, img_height, img_width, img_height])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(img_width - 1, endX), min(img_height - 1, endY))
            
            w = endX - startX
            h = endY - startY
            detected_faces.append((startX, startY, w, h))

    if len(detected_faces) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"message": "Không tìm thấy khuôn mặt trong ảnh."}
        )
        
    if len(detected_faces) > 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"message": "Ảnh chứa nhiều hơn một khuôn mặt. Vui lòng cung cấp ảnh chỉ có một khuôn mặt."}
        )

    # At this point, we know there is exactly one face
    x, y, w, h = detected_faces[0]
    face_area = w * h

    if (face_area / img_area) < request.face_area_threshold:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"message": f"Khuôn mặt quá nhỏ. Kích thước khuôn mặt phải lớn hơn {request.face_area_threshold * 100}% kích thước cả ảnh."}
        )

    # Crop the image to double the size around the face
    # Calculate the center of the face
    center_x = x + w // 2
    center_y = y + h // 2
    
    # Calculate the size of the new crop area (double the face dimensions)
    crop_w = w * 2
    crop_h = h * 2

    # Calculate the top-left corner of the crop area
    crop_x1 = center_x - crop_w // 2
    crop_y1 = center_y - crop_h // 2

    # **Crucially, ensure the crop coordinates do not go out of the image bounds**
    crop_x1 = max(0, crop_x1)
    crop_y1 = max(0, crop_y1)
    
    # Calculate the bottom-right corner based on the adjusted top-left
    crop_x2 = min(img_width, crop_x1 + crop_w)
    crop_y2 = min(img_height, crop_y1 + crop_h)
    
    # Adjust top-left again if the bottom-right hit the edge
    crop_x1 = max(0, crop_x2 - crop_w)
    crop_y1 = max(0, crop_y2 - crop_h)

    # Perform the crop
    cropped_image = image[crop_y1:crop_y2, crop_x1:crop_x2]
    
    # 6. Encode the cropped image back to base64
    _, buffer = cv2.imencode('.jpg', cropped_image)
    cropped_base64 = base64.b64encode(buffer).decode('utf-8')

    # 7. Return the successful response
    return SuccessResponse(img=cropped_base64)
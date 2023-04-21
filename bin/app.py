import base64
from fastapi import FastAPI
from omegaconf import OmegaConf
from pydantic import BaseModel
from predict_b64 import predict as predict_b64

class Payload(BaseModel):
    # timestamp: str
    mask: bytes
    image: bytes

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/predict")
def predict(payload: Payload):

    b64_mask = payload.mask
    b64_image = payload.image


    b64_bytes = base64.b64encode(b64_mask)
    mask_decoded_string = base64.b64decode(b64_bytes)

    b64_bytes = base64.b64encode(b64_image)
    image_decoded_string = base64.b64decode(b64_bytes)

    config = OmegaConf.load('../configs/prediction/default.yaml')
    b64_predit = predict_b64(predict_config=config, b64_image=image_decoded_string, b64_mask=mask_decoded_string,  img_ext='.jpg')
    return b64_predit
import logging
import os
import tempfile

import nibabel as nib
import numpy as np
import torch
import yaml
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src.models.SWIN4D import SWIN4D

config = yaml.safe_load(open("configs/config.yaml", "r"))
app = FastAPI()
model_age = SWIN4D(config)
model_age.load_state_dict(torch.load(config["best_swin_age_group"], map_location=torch.device("cpu")))
model_age.eval()

model_sex = SWIN4D(config)
model_sex.load_state_dict(torch.load(config["best_swin_sex"], map_location=torch.device("cpu")))
model_sex.eval()


#! GET Routes
@app.get("/")
def read_root():
    return {"message": "Welcome to the ADNI dataset Prediction API!"}


@app.get("/health")
def health_check():
    return {"status": "ok", "message": "API is running"}


@app.get("/help")
def get_model_info():
    return {
        "model_name": "SWIN4D",
        "version": "1.0.0",
        "trained_on": "ADNI Dataset",
        "input_shape": config["img_size"],  # Assuming last dim is 20
        "output_classes": ["Age Group (Young, Old), Gender (F, M)"],
    }


#! Helper Functions
class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    raw_output: float
    IntegratedGradients: str


async def load_and_predict(nifti_file, model):

    with tempfile.NamedTemporaryFile(suffix=".nii", delete=False) as tmp:
        try:
            # Save to a temporary file because nibabel needs a file path
            contents = await nifti_file.read()
            tmp.write(contents)

            # Data loading and preprocessing
            fmri_img = nib.load(tmp.name)
            fmri_data = fmri_img.dataobj[:, :, :, 70 : 70 + 20]
            fmri_data = pad_4d(fmri_data)  # Pad to 120x120x120x20
            fmri_data = (fmri_data - fmri_data.min()) / (fmri_data.max() - fmri_data.min() + 1e-8)
            fmri_data = fmri_data.unsqueeze(0).unsqueeze(0)
            fmri_data = fmri_data.to(torch.device("cpu"))

            with torch.no_grad():
                output = model(fmri_data)

            return output

        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"File not found at: {nifti_file.filename}")
        except Exception as e:
            logging.error(f"Prediction error: {e}")
            raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {e}")
        finally:
            if os.path.exists(tmp.name):
                os.remove(tmp.name)
                logging.info(f"Temporary file removed: {tmp.name}")


def pad_4d(fmri_data):
    background_value = fmri_data[0, 0, 0]  # Find background value
    padded_volume = np.full(config["img_size"], background_value, dtype=fmri_data.dtype)

    pad_x = (config["img_size"][0] - fmri_data.shape[0]) // 2
    pad_y = (config["img_size"][1] - fmri_data.shape[1]) // 2
    pad_z = (config["img_size"][2] - fmri_data.shape[2]) // 2

    padded_volume[
        pad_x : pad_x + fmri_data.shape[0], pad_y : pad_y + fmri_data.shape[1], pad_z : pad_z + fmri_data.shape[2]
    ] = fmri_data

    return torch.tensor(padded_volume, dtype=torch.float32)


#! POST Routes
@app.post("/predict_age", response_model=PredictionResponse)
async def predict_age(nifti_file: UploadFile):

    model_output = await load_and_predict(nifti_file, model_age)

    output = model_output.view(-1)  # for BCEWithLogitsLoss
    sigmoid_value = torch.sigmoid(output).item()
    sigmoid_probability = int(sigmoid_value <= 0.5)

    prediction = "Young" if sigmoid_probability else "Old"
    confidence = sigmoid_value if prediction == "Old" else 1 - sigmoid_value
    gradients = f"/results/visualization/swin_2targets_age/ADNI_age_group_target{sigmoid_probability}_tsh10.png"

    return {
        "prediction": prediction,
        "confidence": round(confidence, 3),
        "raw_output": round(output.item(), 3),
        "IntegratedGradients": gradients,
    }


@app.post("/predict_sex", response_model=PredictionResponse)
async def predict_sex(nifti_file: UploadFile):

    model_output = await load_and_predict(nifti_file, model_sex)

    output = model_output.view(-1)  # for BCEWithLogitsLoss
    sigmoid_value = (sigmoid_value <= 0, 5)
    sigmoid_probability = torch.sigmoid(output).item()

    sex = "F" if sigmoid_probability else "M"
    confidence = sigmoid_value if sex == "M" else 1 - sigmoid_value
    gradients = f"results/visualization/swin_2targets_sex/ADNI_sex_target{sigmoid_probability}_10.png"

    return {
        "prediction": sex,
        "confidence": round(confidence, 3),
        "raw_output": round(output.item(), 3),
        "IntegratedGradients": gradients,
    }

import os
import tempfile

import nibabel as nib
import numpy as np
import torch
import yaml
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from src.models.SWIN4D import SWIN4D

config = yaml.safe_load(open("configs/config.yaml", "r"))
app = FastAPI()
model = SWIN4D(config)
model.load_state_dict(torch.load(config["best_swin_age_group"], map_location=torch.device("cpu")))
model.eval()


@app.get("/")
def read_root():
    return {"message": "Welcome to the ADNI dataset Age Group Prediction API!"}


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


class PredictionPath(BaseModel):
    path: str  # one field path of type string


class PredictionResponse(BaseModel):
    age_group: str


@app.post("/predict_age", response_model=PredictionResponse)
def predict_age(fmri_path: PredictionPath):

    try:
        fmri_img = nib.load(fmri_path.path)
        fmri_data = fmri_img.dataobj[:, :, :, 70 : 70 + 20]
        fmri_data = pad_4d(fmri_data)  # Pad to 120x120x120x20
        fmri_data = (fmri_data - fmri_data.min()) / (fmri_data.max() - fmri_data.min() + 1e-8)
        fmri_data = fmri_data.unsqueeze(0).unsqueeze(0)
        fmri_data = fmri_data.to(torch.device("cpu"))

        with torch.no_grad():
            output = model(fmri_data)

        output = output.view(-1)  # for BCEWithLogitsLoss
        predicted_labels = (torch.sigmoid(output) >= 0.5).long()

        age_group = "Young" if predicted_labels == 0 else "Old"

        return {"age_group": age_group}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"File not found at: {fmri_path}")
    except Exception as e:
        print(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {e}")


@app.post("/predict_age_default", response_model=PredictionResponse)
def predict_age_default():
    fmri_path = PredictionPath(path="API/fmri.nii")
    return predict_age(fmri_path)


@app.post("/predict_niftii")
async def predict_niftii(nifti_file: UploadFile):

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

            output = output.view(-1)  # for BCEWithLogitsLoss
            predicted_labels = (torch.sigmoid(output) >= 0.5).long()

            age_group = "Young" if predicted_labels == 0 else "Old"
            return {"age_group": age_group}

        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"File not found at: {nifti_file.filename}")
        except Exception as e:
            print(f"Prediction error: {e}")
            raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {e}")
        finally:
            if os.path.exists(tmp.name):
                os.remove(tmp.name)
                print(f"Temporary file removed: {tmp.name}")

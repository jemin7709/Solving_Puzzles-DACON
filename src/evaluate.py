import json
from datetime import datetime, timedelta

import pandas as pd
import torch
from dataset.data_loader import CustomDataset
from models.unet import BaseModel
from torch.utils.data import DataLoader
from torchvision import transforms
from trainer.trainer import inference
from utils.util import seed_everything

### Set Time ###

current_time_kst = datetime.utcnow() + timedelta(hours=9)
formatted_time = current_time_kst.strftime("%Y-%m-%d-%H:%M:%S")

### Set Device ###

device = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
)
print(device)

### Set Config and Fix seed ###

with open("configs/config.json", "r") as f:
    CFG = json.load(f)

seed_everything(CFG["SEED"])

### Load Data ###

test_df = pd.read_csv(CFG["PATH"] + "test.csv")

test_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((CFG["IMG_SIZE"], CFG["IMG_SIZE"])),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
test_dataset = CustomDataset(test_df["img_path"].values, None, test_transform)
test_loader = DataLoader(
    test_dataset,
    batch_size=CFG["BATCH_SIZE"],
    shuffle=False,
    num_workers=CFG["NUM_WORKERS"],
)

### Load Model ###

model = BaseModel().to(device)
model_state_dict = torch.load("weights/model_state_dict.pt", map_location=device)
model.load_state_dict(model_state_dict)

### Inference ###

preds = inference(model, test_loader, device)

submit = pd.read_csv("files/sample_submission.csv")
submit.iloc[:, 1:] = preds
submit.iloc[:, 1:] += 1
submit.to_csv(f"output/{formatted_time}_baseline_submit.csv", index=False)

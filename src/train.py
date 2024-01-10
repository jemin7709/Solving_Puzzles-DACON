import json

import pandas as pd
import torch
from dataset.data_loader import CustomDataset
from models.unet import BaseModel
from torch.utils.data import DataLoader
from torchvision import transforms
from trainer.trainer import train
from utils.util import seed_everything

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

train_df = pd.read_csv(CFG["PATH"] + "train.csv")
train_len = int(len(train_df) * 0.7)

train_df = train_df.iloc[:train_len]
val_df = train_df.iloc[train_len:]

train_labels = train_df.iloc[:, 2:].values.reshape(-1, 4, 4)
val_labels = val_df.iloc[:, 2:].values.reshape(-1, 4, 4)

train_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((CFG["IMG_SIZE"], CFG["IMG_SIZE"])),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
test_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((CFG["IMG_SIZE"], CFG["IMG_SIZE"])),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

train_dataset = CustomDataset(
    train_df["img_path"].values, train_labels, train_transform
)
train_loader = DataLoader(
    train_dataset,
    batch_size=CFG["BATCH_SIZE"],
    shuffle=True,
    num_workers=CFG["NUM_WORKERS"],
)

val_dataset = CustomDataset(val_df["img_path"].values, val_labels, test_transform)
val_loader = DataLoader(
    val_dataset, batch_size=CFG["BATCH_SIZE"], shuffle=False, num_workers=0
)

### Load Model ###
model = BaseModel()
optimizer = torch.optim.Adam(params=model.parameters(), lr=CFG["LEARNING_RATE"])
infer_model = train(model, optimizer, train_loader, val_loader, device, CFG)

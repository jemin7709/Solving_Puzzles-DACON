import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


def validation(model, criterion, val_loader, device):
    model.eval()
    val_loss = []
    val_acc = []

    with torch.no_grad():
        for imgs, labels in tqdm(iter(val_loader)):
            imgs = imgs.float().to(device)
            labels = labels.to(device)

            output = model(imgs)

            loss = criterion(output, labels)

            val_loss.append(loss.item())

            # 정확도 계산을 위한 예측 레이블 추출
            predicted_labels = torch.argmax(output, dim=1)

            # 샘플 별 정확도 계산
            for predicted_label, label in zip(predicted_labels, labels):
                val_acc.append(((predicted_label == label).sum() / 16).item())

        _val_loss = np.mean(val_loss)
        _val_acc = np.mean(val_acc)

    return _val_loss, _val_acc


def train(model, optimizer, train_loader, val_loader, device, CFG):
    model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    best_val_acc = 0
    best_model = None
    for epoch in range(1, CFG["EPOCHS"] + 1):
        model.train()
        train_loss = []
        for imgs, labels in tqdm(iter(train_loader)):
            imgs = imgs.float().to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            output = model(imgs)
            loss = criterion(output, labels)

            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

        _val_loss, _val_acc = validation(model, criterion, val_loader, device)
        _train_loss = np.mean(train_loss)
        print(
            f"Epoch [{epoch}], Train Loss : [{_train_loss:.5f}] Val Loss : [{_val_loss:.5f}] Val ACC : [{_val_acc:.5f}]"
        )

        if best_val_acc < _val_acc:
            best_val_acc = _val_acc
            best_model = model
            torch.save(best_model.state_dict(), "weights/model_state_dict.pt")

    return best_model


def inference(model, test_loader, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for imgs in tqdm(iter(test_loader)):
            imgs = imgs.float().to(device)

            output = model(imgs)

            # 정확도 계산을 위한 예측 레이블 추출
            predicted_labels = torch.argmax(output, dim=1).view(-1, 16)
            predicted_labels = predicted_labels.cpu().detach().numpy()

            preds.extend(predicted_labels)

    return preds

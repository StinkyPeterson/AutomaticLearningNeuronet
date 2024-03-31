
import shutil
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from functions import get_train_augs,get_val_augs,train_model,eval_model
from SegmentationModel import SegmentationModel
from SegmentationDataset import SegmentationDataset
from io import BytesIO
import socketio
from sanic import Sanic
from sanic.response import file

import base64
import os
import zipfile
import os
import asyncio

stop_training = False
training_lock = asyncio.Lock()

MAX_REQUEST_SIZE = 100 * 1024 * 1024  # 100 МБ

sio = socketio.AsyncServer(async_mode='sanic', cors_allowed_origins='*', ping_timeout = 300, max_http_buffer_size = MAX_REQUEST_SIZE)
app = Sanic(__name__)
sio.attach(app)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ENCODER = 'timm-efficientnet-b0'
WEIGHTS = 'imagenet'
UPLOAD_FOLDER = 'uploaded_files'
training_threads = {}
model_counter = 1
modeles = [
    {
        "id": 0,
        "value": "Unet"
    },
        {
        "id": 1,
        "value": "UnetPlusPlus"
    },
        {
        "id": 2,
        "value": "MAnet"
    },
        {
        "id": 3,
        "value": "Linknet"
    },
        {
        "id": 4,
        "value": "FPN"
    },
        {
        "id": 5,
        "value": "PSPNet"
    },
            {
        "id": 6,
        "value": "DeepLabV3"
    },
            {
        "id": 7,
        "value": "DeepLabV3Plus"
    },
            {
        "id": 8,
        "value": "PAN"
    },
    ]

@sio.event
async def connect(sid, environ):
    print(f"Connected: {sid}")
    # if sio.recovered:
    #     print("Соединение восстановлено")
    # else:
    #     print('Новое подключение')

@sio.event
async def disconnect(sid):
    print(f"Disconnected: {sid}")

@sio.event
async def message(sid, data):
    print(f"Message from {sid}: {data}")
    await sio.emit('response', data="Received your message!", room=sid)

@sio.event
async def get_models(sid):
    await sio.emit('send_models', data = modeles, room=sid)
    
@sio.event
async def start(sid, data):
    await start_training("dataset/train.csv", data["eraCount"], 0.001, 224, data["partitionLevel"], data["idModel"], data["validationPercent"], data['isAutomaticStop'], sid)

@sio.event
async def chunk(sid, data):
    os.makedirs(f'DataSets/{sid}/dataset', exist_ok=True)
    with open(f'DataSets/{sid}/dataset/output.zip', 'ab') as f:
        f.write(data)
    print(f'Received chunk of size {len(data)} bytes')

@sio.event
async def unpacking_dataset(sid):
    print('Received end signal')
    print('File received successfully')

    # После получения всех чанков архива, распаковываем его
    try:
        with zipfile.ZipFile(f"DataSets/{sid}/dataset/output.zip", 'r') as zip_ref:
            zip_ref.extractall(f"DataSets/{sid}/dataset")
        print('Archive unpacked successfully')
    except Exception as e:
        print('Error unpacking archive:', e)

    # Опционально - удаляем архив после распаковки, если он больше не нужен
    try:
        os.remove(f"DataSets/{sid}/dataset/output.zip")
        print('Archive removed successfully')
    except Exception as e:
        print('Error removing archive:', e)
    await sio.emit("dataset_loaded")

@sio.event
async def download_file(sid):
    file_path = f'Saves/{sid}/best_model.pt'  # Путь к вашему файлу на сервере

    with open(file_path, 'rb') as f:
        file_data = f.read()

    # Кодируем данные файла в base64
    file_base64 = base64.b64encode(file_data).decode('utf-8')

    await sio.emit('file_data', file_base64, room=sid)


def delete_everything_in_folder(folder_path):
    shutil.rmtree(folder_path)
    os.mkdir(folder_path)

async def start_training(TRAIN_DATA_PATH, EPOCHS, LR, IMG_SIZE, BATCH_SIZE, MODEL, TEST_SIZE, isSTOP, sid):
    DATA_DIR = 'Datasets/' + sid + '/'
    SAVE_DIR = 'Saves/' + sid + '/'
    os.makedirs(SAVE_DIR, exist_ok=True)
    global DEVICE, ENCODER, WEIGHTS, training_threads

    global stop_training
    async with training_lock:
        stop_training = False

    # Это читка файла разметки
    df = pd.read_csv(DATA_DIR + TRAIN_DATA_PATH)
    train_df, val_df = train_test_split(df, test_size=TEST_SIZE, random_state=57)

    train_data = SegmentationDataset(train_df, get_train_augs(IMG_SIZE), DATA_DIR)
    val_data = SegmentationDataset(val_df, get_val_augs(IMG_SIZE), DATA_DIR)

    trainloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    valloader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)

    model = SegmentationModel(MODEL, ENCODER, WEIGHTS)
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    best_val_loss = 1e9

    for i in range(1, EPOCHS + 1):
        train_loss, train_dice = train_model(trainloader, model, optimizer, DEVICE)
        val_loss, val_dice = eval_model(valloader, model, DEVICE)
        # print(training_threads)
        # if training_threads[sid]:
        #     break
        async with training_lock:
            if stop_training:
                break

        if val_loss < best_val_loss:
            torch.save(model.state_dict(), SAVE_DIR + "best_model.pt")
            print("MODEL SAVED")
            best_val_loss = val_loss

        elif (isSTOP and i>3):            
            break
        

        dataDiagram = {
            "epoch": i,
            "trainLossY": train_loss,
            "trainDiceY": train_dice,
            "valLossY": val_loss,
            "valDiceY": val_dice
        }
        await sio.emit("send_diagram", dataDiagram, sid)
        print(f"\033[1m\033[92m Epoch {i} Train Loss {train_loss} Train dice {train_dice} Val Loss {val_loss} Val Dice {val_dice}")

    print(f"Model {sid} training completed.")
    await sio.emit('end_training', room=sid)

def send_data(dataDiagram, sid):
    print("send_data", sid)
    sio.emit("send_diagram", dataDiagram, room=sid)



def send_data(dataDiagram, sid):
    print("send_data", sid)
    sio.emit("send_diagram", dataDiagram, room=sid)


if __name__ == "__main__":
    app.run(host="localhost", port=8765)
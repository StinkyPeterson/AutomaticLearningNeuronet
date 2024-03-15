
import random
#from main import modeles, start_training
import threading
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
from sanic import Sanic, response

import socketio
import base64
import os
import zipfile
import os

MAX_REQUEST_SIZE = 100 * 1024 * 1024  # 100 МБ

sio = socketio.AsyncServer(async_mode='sanic', cors_allowed_origins='*', ping_timeout = 300, max_http_buffer_size = MAX_REQUEST_SIZE)
app = Sanic(__name__)
sio.attach(app)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ENCODER = 'timm-efficientnet-b0'
WEIGHTS = 'imagenet'
UPLOAD_FOLDER = 'uploaded_files'
locker = threading.Lock()
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

@sio.event
async def disconnect(sid):
    print(f"Disconnected: {sid}")

@sio.event
async def message(sid, data):
    print(f"Message from {sid}: {data}")
    await sio.emit('response', data="Received your message!", room=sid)

@sio.event
async def get_models(sid):
    print("ОТПРАВКА МОДЕЛЕЙ КЛИЕНТУ")
    await sio.emit('send_models', data = modeles, room=sid)
    
@sio.event
async def start(sid, data):
    print()
    await start_training("dataset/train.csv", data["eraCount"], 0.001, 224, data["partitionLevel"], data["idModel"], data["validationPercent"], sid)



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



def delete_everything_in_folder(folder_path):
    shutil.rmtree(folder_path)
    os.mkdir(folder_path)
async def start_training(TRAIN_DATA_PATH, EPOCHS, LR, IMG_SIZE, BATCH_SIZE, MODEL, TEST_SIZE, sid):
    DATA_DIR = 'Datasets/' + sid + '/'
    SAVE_DIR = 'Saves/' + sid + '/'
    os.makedirs(SAVE_DIR, exist_ok=True)
    global DEVICE, ENCODER, WEIGHTS

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

        if val_loss < best_val_loss:
            torch.save(model.state_dict(), SAVE_DIR + "best_model.pt")
            print("MODEL SAVED")
            best_val_loss = val_loss

        dataDiagram = {
            "epoch": i,
            "trainLossY": train_loss,
            "trainDiceY": train_dice,
            "valLossY": val_loss,
            "valDiceY": val_dice
        }
        await sio.emit("send_diagram", dataDiagram, sid)
        # t = threading.Thread(target=send_data, args=(dataDiagram, sid))
        # t.start()
        print(f"\033[1m\033[92m Epoch {i} Train Loss {train_loss} Train dice {train_dice} Val Loss {val_loss} Val Dice {val_dice}")

    print(f"Model {sid} training completed.")
    await sio.emit('end_training', room=sid)

def send_data(dataDiagram, sid):
    print("send_data", sid)
    sio.emit("send_diagram", dataDiagram, room=sid)


def train_thread(TRAIN_DATA_PATH,EPOCHS,LR,IMG_SIZE,BATCH_SIZE,MODEL,TEST_SIZE, sid):
    DATA_DIR = 'Datasets/' + "1" + '/'
    SAVE_DIR = 'Saves/' + "1" + '/'
    global DEVICE, ENCODER, WEIGHTS
    sio.emit("response", "НАЧАЛО ОБУЧЕНИЯ", room=sid)

    #Это читка файла разметки
    df = pd.read_csv(DATA_DIR + TRAIN_DATA_PATH)
    #locker.acquire()
    # print(df.shape)
    # print(df.head(10))
    #locker.release()

    # Разбитие на тестовую и валидационную выборку
    train_df, val_df = train_test_split(df, test_size=TEST_SIZE, random_state=57)

    train_data = SegmentationDataset(train_df, get_train_augs(IMG_SIZE), DATA_DIR)
    val_data = SegmentationDataset(val_df, get_val_augs(IMG_SIZE), DATA_DIR)
    #locker.acquire()
    # print(f"Size of Trainset : {len(train_data)}")
    # print(f"Size of Validset : {len(val_data)}")
    #locker.release()

    #Это подгрузка изображений по группам
    trainloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    valloader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)
    #locker.acquire()
    # print(f"Total number of batches in Train Loader: {len(trainloader)}")
    # print(f"Total number of batches in Val Loader: {len(valloader)}")
    #locker.release()

    #Это объявление модели
    # print(type(MODEL))
    # print(MODEL)
    model = SegmentationModel(MODEL, ENCODER, WEIGHTS)
    model.to(DEVICE)
    # print(model.parameters(),ENCODER,WEIGHTS)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    best_val_loss = 1e9

    # Цикл обучения по эпохам
    for i in range(1, EPOCHS + 1):
        train_loss, train_dice = train_model(trainloader, model, optimizer, DEVICE)
        val_loss, val_dice = eval_model(valloader, model, DEVICE)
        #Если в этот цикл потерь меньше - сохраняем
        if val_loss < best_val_loss:
            # Save the best model
            torch.save(model.state_dict(), SAVE_DIR + "best_model.pt")
            print("MODEL SAVED")

            best_val_loss = val_loss
        #Иначе - обрубаем обучение
        #

        #Вот это итоги эпохи
        ##locker.acquire()
        dataDiagram = {
            "epoch": i,
            "trainLossY": train_loss,
            "trainDiceY": train_dice,
            "valLossY": val_loss,
            "valDiceY": val_dice
        }
        t = threading.Thread(target=send_data(), args=(dataDiagram, sid))
        t.start()
        print(f"\033[1m\033[92m Epoch {i} Train Loss {train_loss} Train dice {train_dice} Val Loss {val_loss} Val Dice {val_dice}")
        ##locker.release()

    print(f"Model {sid} training completed.")
    sio.emit('end_training', room=sid)

def send_data(dataDiagram, sid):
    print("send_data", sid)
    sio.emit("send_diagram", dataDiagram, room=sid)


if __name__ == "__main__":
    app.run(host="localhost", port=8765)
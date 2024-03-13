
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
import asyncio
import base64
from io import BytesIO
import zipfile

import socketio
sio = socketio.Server(cors_allowed_origins='http://localhost:3000')
app = socketio.WSGIApp(sio)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ENCODER = 'timm-efficientnet-b0'
WEIGHTS = 'imagenet'
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
def connect(sid, environ):
    print(f"Connected: {sid}")

@sio.event
def disconnect(sid):
    print(f"Disconnected: {sid}")

@sio.event
def message(sid, data):
    print(f"Message from {sid}: {data}")
    sio.emit('response', data="Received your message!", room=sid)

@sio.event
def get_models(sid):
    sio.emit('send_models', data = modeles, room=sid)
    sio.emit('response', data="Received your message!", room=sid)

@sio.event
def force_stop(sid):
    print('Принудительная остановка')
    
@sio.event
def start(sid, data):
    print(data)
    # sio.start_background_task(start_training, args = ("dataset/train.csv", data["eraCount"], 0.001, 224, data["partitionLevel"], data["idModel"], data["validationPercent"], sid))
    sio.start_background_task(target= lambda: start_training("dataset/train.csv", data["eraCount"], 0.001, 224, data["partitionLevel"], data["idModel"], data["validationPercent"], sid))

def delete_everything_in_folder(folder_path):
    shutil.rmtree(folder_path)
    os.mkdir(folder_path)
def start_training(TRAIN_DATA_PATH, EPOCHS, LR, IMG_SIZE, BATCH_SIZE, MODEL, TEST_SIZE, sid):
    DATA_DIR = 'Datasets/' + "1" + '/'
    SAVE_DIR = 'Saves/' + "1" + '/'
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
        t = threading.Thread(target=send_data, args=(dataDiagram, sid))
        t.start()
        print(f"\033[1m\033[92m Epoch {i} Train Loss {train_loss} Train dice {train_dice} Val Loss {val_loss} Val Dice {val_dice}")

    print(f"Model {sid} training completed.")
    sio.emit('end_training', room=sid)

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


if __name__ == '__main__':
    import eventlet
    # eventlet.monkey_patch()
    eventlet.wsgi.server(eventlet.listen(('', 5000)), app)
import socketio
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
    print("Получание моделей", sid)
    sio.emit('send_models', data = modeles, room=sid)
    sio.emit('response', data="Received your message!", room=sid)

# def send_data_periodically(sid):
#     while True:
#         data = random.randint(1, 10)
#         sio.emit('periodic_data', data, room=sid)
#         eventlet.sleep(1)  # Подождать 1 секунду перед отправкой следующего сообщения

# @sio.event
# def start_periodic_data(sid):
#     print(f"Starting periodic data to {sid}")
#     eventlet.spawn(send_data_periodically, sid)

@sio.event
def force_stop(sid):
    print('Принудительная остановка')
    
@sio.event
def start(sid, data):
    print(data)
    eventlet.spawn(start_training, "dataset/train.csv", data["eraCount"], 0.001, 224, data["partitionLevel"], data["idModel"], data["validationPercent"], sid)
    # start_training("dataset/train.csv", data["eraCount"], 0.001, 224, data["partitionLevel"], data["idModel"], data["validationPercent"], sid)


def delete_everything_in_folder(folder_path):
    shutil.rmtree(folder_path)
    os.mkdir(folder_path)
def start_training(TRAIN_DATA_PATH,EPOCHS,LR,IMG_SIZE,BATCH_SIZE,MODEL,TEST_SIZE, sid):
    print("start_training", sid)
    sio.emit("response", "start_training", room=sid)
    global model_counter
    model_name = model_counter
    model_counter += 1
    # train_thread(TRAIN_DATA_PATH,EPOCHS,LR,IMG_SIZE,BATCH_SIZE,MODEL,TEST_SIZE, sid)
    # t = threading.Thread(target=train_thread, args=(TRAIN_DATA_PATH,EPOCHS,LR,IMG_SIZE,BATCH_SIZE,MODEL,TEST_SIZE, sid, sio))
    # t.name=str(model_name)
    # t.start()
    # training_threads[model_name] = t
    # return f"Training Model {model_name} started."
    sio.start_background_task(target= lambda: train_thread(TRAIN_DATA_PATH,EPOCHS,LR,IMG_SIZE,BATCH_SIZE,MODEL,TEST_SIZE, sid))
    # try:
    #     t = threading.Thread(target=train_thread, args=(TRAIN_DATA_PATH,EPOCHS,LR,IMG_SIZE,BATCH_SIZE,MODEL,TEST_SIZE, sid))
    #     t.name=str(model_name)
    #     t.start()
    #     training_threads[model_name] = t
    #     return f"Training Model {model_name} started."
    # except Exception as e:
    #     print(f"An error occurred while starting training: {e}")
    #     # Handle the error if needed
    #     return f"Error starting training for Model {model_name}"

def train_thread(TRAIN_DATA_PATH,EPOCHS,LR,IMG_SIZE,BATCH_SIZE,MODEL,TEST_SIZE, sid):
    #locker.acquire()
    print("НАЧАЛО ОБУЧЕНИЯ")
    sio.emit("response", "НАЧАЛО ОБУЧЕНИЯ", room=sid)
    # print(sid)
    print(TRAIN_DATA_PATH, EPOCHS, BATCH_SIZE, MODEL)
    #locker.release()
    
    
    DATA_DIR = 'Datasets/' + "1" + '/'
    SAVE_DIR = 'Saves/' + "1" + '/'
    # DATA_DIR = 'Datasets/' + "1" + '/'
    # SAVE_DIR = 'Saves/' + "1" + '/'
    global DEVICE, ENCODER, WEIGHTS

    #Это читка файла разметки
    df = pd.read_csv(DATA_DIR + TRAIN_DATA_PATH)
    #locker.acquire()
    print(df.shape)
    print(df.head(10))
    #locker.release()

    # Разбитие на тестовую и валидационную выборку
    train_df, val_df = train_test_split(df, test_size=TEST_SIZE, random_state=57)

    train_data = SegmentationDataset(train_df, get_train_augs(IMG_SIZE), DATA_DIR)
    val_data = SegmentationDataset(val_df, get_val_augs(IMG_SIZE), DATA_DIR)
    #locker.acquire()
    print(f"Size of Trainset : {len(train_data)}")
    print(f"Size of Validset : {len(val_data)}")
    #locker.release()

    #Это подгрузка изображений по группам
    trainloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    valloader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)
    #locker.acquire()
    print(f"Total number of batches in Train Loader: {len(trainloader)}")
    print(f"Total number of batches in Val Loader: {len(valloader)}")
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
        # print(dataDiagram, sid)
        sio.emit("send_diagram", dataDiagram, room = sid)
        # try:
        #     print("Отправка сообщения", sid)
        #     sio.emit("response", "работай сука", room=sid)
        # except Exception as e:
        #     print(f"An error occurred while emitting response: {e}")
        print(f"\033[1m\033[92m Epoch {i} Train Loss {train_loss} Train dice {train_dice} Val Loss {val_loss} Val Dice {val_dice}")
        ##locker.release()

    #locker.acquire()
    # print(f"Model {threading.current_thread().name} training completed.")
    #locker.release()
    print(f"Model {1} training completed.")



if __name__ == '__main__':
    import eventlet
    eventlet.monkey_patch()
    eventlet.wsgi.server(eventlet.listen(('', 5000)), app)
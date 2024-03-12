import socketio
import random

sio = socketio.Server(cors_allowed_origins='http://localhost:3000')
app = socketio.WSGIApp(sio)

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
    print("Получание моделей")
    models = [
        {
            "id": 0,
            "value": "нейросеть_name_1"
        },
        {
            "id": 1,
            "value": "нейросеть_name_2"
        },
        {
            "id": 2,
            "value": "нейросеть_name_3"
        }
    ]

    sio.emit('send_models', data = models, room=sid)

@sio.event
def send_diagram(sid):
    sio.emit('send_diagram')


def send_data_periodically(sid):
    while True:
        data = random.randint(1, 10)
        sio.emit('periodic_data', data, room=sid)
        eventlet.sleep(1)  # Подождать 1 секунду перед отправкой следующего сообщения

@sio.event
def start_periodic_data(sid):
    print(f"Starting periodic data to {sid}")
    eventlet.spawn(send_data_periodically, sid)

@sio.event
def force_stop(sid):
    print('Принудительная остановка')
    


if __name__ == '__main__':
    import eventlet
    eventlet.wsgi.server(eventlet.listen(('', 5000)), app)

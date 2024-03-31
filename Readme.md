## Инструкция по развертыванию

## Установка

Склонируйте репозиторий на свой компьютер:
```bash
git clone https://github.com/StinkyPeterson/AutomaticLearningNeuronet.git
```

### Требования:

Установите следующие программное обеспечение:

1. [Docker Desktop](https://www.docker.com).
2. [Git](https://github.com/git-guides/install-git).

### Инструкция установки

1. Соберите контейнеры докера с помощью команды:
    ```shell
    docker compose build
    ```
   
2. To start the application run:
    ```shell
    docker compose up
    ```
    Когда контейнеры будут запущены, сервер запустится по адресу http://0.0.0.0:80. Вы можете открыть его в своем браузере.
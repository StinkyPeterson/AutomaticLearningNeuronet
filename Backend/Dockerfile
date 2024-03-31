# Используем официальный образ Python версии 3.8
FROM python:3.8

# Устанавливаем переменную окружения PYTHONUNBUFFERED для предотвращения буферизации вывода
ENV PYTHONUNBUFFERED 1

# Создаем и устанавливаем рабочую директорию внутри Docker контейнера
WORKDIR /app

# Копируем файл зависимостей и устанавливаем их
COPY requirements.txt /app/
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Копируем все содержимое текущей директории в /app внутри Docker контейнера
COPY . /app/

# Определяем порт, который будет открыт в контейнере
EXPOSE 8000

# Запускаем команду для запуска приложения (замените на свою команду)
CMD ["python", "server.py"]
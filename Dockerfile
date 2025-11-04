FROM python:3.12-slim-bullseye

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONPATH=/app:$PYTHONPATH

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsm6 libxext6 libgl1-mesa-glx \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY facetools ./facetools
COPY data ./data

ENV FLASK_APP=app/app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=5000

EXPOSE 5000

CMD ["flask", "run"]

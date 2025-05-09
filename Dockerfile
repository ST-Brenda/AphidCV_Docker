FROM python:3.10-slim

WORKDIR /application

# Instala dependências do OpenCV 
RUN apt-get update && \
    apt-get install -y \
    libopencv-dev \
    && apt-get clean
    
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip  && \
    pip install dill && \
    pip install --no-cache-dir -r requirements.txt --timeout=1000

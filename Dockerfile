FROM python:3.10-slim

WORKDIR /application

# Instala dependências do OpenCV 
RUN apt-get update && \
    apt-get install -y \
    libopencv-dev \
    && apt-get clean
    
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt --timeout=1000

# # Copia os diretórios e arquivos necessários
# # Não está sendo usado, pois nesta versão estão sendo compartilhadas as pastas do host ao rodar um container
# COPY yolov8_cgpuhead_detect.py .
# COPY modelos/ modelos/
# COPY imagens/ imagens/

# Executa o script Python
ENTRYPOINT ["python", "yolov8_cgpuhead_detect.py"]
                                                                 

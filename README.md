# 🪲 AphidCV - YOLOv8 Detect Container

Esta imagem Docker é capaz de realizar a inferência em imagens de afídeos utilizando modelos YOLOv8 personalizados.  
A versão `2.0` foi construída com suporte à montagem de volumes no host, permitindo maior flexibilidade para testes e ajustes no script ou modelos.

---

## Instalar a imagem via Docker Hub

Para usar a imagem do container diretamente do Docker Hub, você pode fazer o **pull** da imagem com o seguinte comando:

```bash
docker pull brendaslongotaca/script_detect:2.0
```

---

## 🗂️ Estrutura esperada do diretório no host

No diretório onde o container será executado, é necessário manter a seguinte estrutura:

```
.
├── imagens/                  # Contém as imagens para realizar a inferência 
├── modelos/                  # Contém os modelos personalizados YOLOv8 (.pt)
├── requirements.txt          # Lista de dependências do Python
└── yolov8_cgpuhead_detect.py # Script que realiza a inferência
```
A pasta "imagens" contém uma imagem de afídeos para exemplo, adicione suas próprias imagens na pasta.
Você pode clonar o presente repositório para obter o script da inferência e o arquivo com as dependências do Python, através do comando:

```bash
git clone https://github.com/ST-Brenda/AphidCV_Docker.git
```

---

## Execução do container

### Rodar a inferência em uma imagem:

```bash
docker run -it --rm \
  -v "$PWD/modelos:/application/modelos" \
  -v "$PWD/imagens:/application/imagens" \
  -v "$PWD/yolov8_cgpuhead_detect.py:/application/yolov8_cgpuhead_detect.py" \
  brendaslongotaca/script_detect:2.0 imagens/<nome_da_imagem>.jpeg --especie <rp|sg|md|sa|mp|bb> --contrast <Float> --brightness <Int>
```
Os resultados são salvos em uma pasta com o nome da imagem, dentro da pasta "imagens".



---

## Acessar o container via bash

Para acessar o shell do container internamente:

```bash
docker run -it \
  --entrypoint bash \
  -v "$PWD/modelos:/application/modelos" \
  -v "$PWD/imagens:/application/imagens" \
  -v "$PWD/yolov8_cgpuhead_detect.py:/application/yolov8_cgpuhead_detect.py" \
  brendaslongotaca/script_detect:2.0
```

---

## 📝 Observações

- A imagem espera que os volumes estejam corretamente montados para encontrar os arquivos necessários.
- Use `--rm` para containers temporários (descartáveis).
- Certifique-se de que os arquivos estejam no caminho correto e com permissões adequadas.

---

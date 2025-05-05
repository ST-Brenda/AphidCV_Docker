import cv2 as cv
import os
import shutil
import csv
# Aphid Classifier
import numpy as np
from datetime import datetime
import json
from ultralytics import YOLO
import argparse


labels = ['winged', 'wingless', 'false', 'nymph']

DIV = 32 # Constante para operar com imagem menor em HoughCircles, e evitar erro de memória

CONFIDENCE = 0.80000

def clean_image(CAMINHO, especie, circle, valcircle, threshold_value):
    print(f'thresh: {threshold_value} ({type(threshold_value)})')


#=================================================================
    print(f"Caminho da imagem: '{CAMINHO}'")
    img = cv.imread(CAMINHO)
    if img is None:
        print(f"ERRO: CAMINHO DA IMAGEM: '{CAMINHO}'")
        exit(1)
#=================================================================

    img = cv.imread(CAMINHO)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (33, 33), 0)
    rows, cols = gray.shape
    print(f'type img: {img.dtype} - type blur: {blur.dtype}')

    #cv.imwrite("y.jpg", blur)

    # PROBLEMA RESOLVIDO (12/08/2021): o resize não estava mantendo a razão proporcional correta,
    # e o algoritmo HoughCircles se perdia.
    # Mantive a ideia de reduzir a imagem na razão 1/32 (constante DIV)
    # O objetivo é permitir que o método HoughCircles execute sem dar erro de memória
    resized = cv.resize(blur, None, fx=0.03125, fy=0.03125, interpolation = cv.INTER_AREA)
    _, width = resized.shape    # height não é usado

    try:
        mask = np.zeros((rows, cols), np.uint8)

        # se a detecção foi feita manualmente
        if circle=='1':
            cv.circle(mask, (int(valcircle[0]), int(valcircle[1])), int(valcircle[2]), (255, 255, 255), -1)
            r = int(valcircle[2]) # raio
            x = int(valcircle[0]) # centro x
            y = int(valcircle[1]) # centro y
            
        else:

            # dp = 2 para melhorar um pouco a sensibilidade
            # minDist = width pega só o maior círculo (distância mínima entre dois círculos é a largura da imagem)
            # minRadius = int(width/8) define raio mínimo de 1/8 da largura da imagem (evita buscar círculos pequenos)
            circles = cv.HoughCircles(resized, cv.HOUGH_GRADIENT, 2, width, minRadius=int(width/8),maxRadius=0)

            circles = np.uint16(np.around(circles))

            # Multiplica-se por DIV para considerar o resize feito a partir da imagem original
            r = circles[0, 0, 2]*DIV # raio
            x = circles[0, 0, 0]*DIV # centro x
            y = circles[0, 0, 1]*DIV # centro y

            print("===================")
            print("Círculo encontrado!")
            print("===================")


            for i in circles[0, :]:
                # Draw on mask
                cv.circle(mask, (i[0]*DIV, int(i[1]*DIV)), int(i[2]*DIV), (255, 255, 255), -1)

        crop_original = img[y-r:y+r, x-r:x+r]
        crop_img = gray[y-r:y+r, x-r:x+r]
        crop_mask = mask[y-r:y+r, x-r:x+r]

        # realiza o crop da imagem com a máscara
        crop_img = cv.bitwise_and(crop_img, crop_img, mask=crop_mask)
        crop_original = cv.bitwise_and(crop_original, crop_original, mask=crop_mask)
        _, crop_th = cv.threshold(crop_img, threshold_value, 255, cv.THRESH_BINARY)

    except:
        print('Erro: provavel falha ao encontrar circulo!')
        exit(1)

    return [crop_original, crop_th, r]


class result:

    def __init__(self, specie, classtype, probability, length, weight, perimeter, area):
        self.specie = specie
        self.classtype = classtype
        self.probability = probability
        self.length = length
        self.weight = weight
        self.perimeter = perimeter
        self.area = area
        
def count_object(IMAGE, THRESH, PXMM, CAMINHO, FILE, especie, minSize, maxSize, maxNymph, thresh, diameter, contrast, brightness, manual_Nymphs, manual_Wingless, manual_Wingeds, manual_Total, modelo_yolo):
    nomeEspecie = {
    'rp': 'Rhopalosiphum padi',
    'sg': 'Schizaphis graminum',
    'md': 'Metopolophium dirhodum',
    'sa': 'Sitobion avenae',
    'mp': 'Myzus persicae',
    'bb': 'Brevicoryne brassicae'}
    
    #PXMM = 45
    print(f'Diretorio: {os.getcwd()}')
    print(f'minSize: {minSize}, maxSize: {maxSize}, maxNymph: {maxNymph}')
    print(f"Caminho: '{CAMINHO}'")

    # Load a yolo model
    #model = YOLO("modelos/rp.pt")  # pretrained model

    model = modelo_yolo

    countWinged = 0
    countWingless = 0
    countNymphs = 0
    countTotal = 0
    totalWeight = 0
    validContours = []
    results = []
    nymphs = []
    wingless = []
    wingeds = []
    contours, _ = cv.findContours(THRESH, cv.RETR_TREE, cv.CHAIN_APPROX_TC89_KCOS)

    
    string = str(datetime.now()).split(' ')[1].split(':')
    pasta = FILE + '-' + str(datetime.now()).split(' ')[0] + '-' + '-'.join(string)

    folder_path = os.path.join(pasta)
    print(f"CAMINHO PASTA: '{folder_path}' ===============")

#==============================================================
    os.mkdir(folder_path)
    os.mkdir(os.path.join(folder_path, 'temp'))
    os.mkdir(os.path.join(folder_path, 'doubtful'))

    for cnt in contours:
        M = cv.moments(cnt)

        if(float(M['m00']) > float((PXMM*minSize)**2) and float(M['m00']) < float((PXMM*maxSize)**2)):
            x, y, w, h = cv.boundingRect(cnt)

            if((w > 2.3*h) or (h > 2.3*w)):
                continue

            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            validContours.append(cnt)

            offset = 22  # Codigo do Elison

            img = IMAGE[int((cy-h/2)-offset):int((cy+h/2)+offset),
                        int((cx-w/2)-offset):int((cx+w/2)+offset)]
            noImage = False
            for v in img.shape:
                if not v:
                    noImage = True
            if noImage:
                continue
            
            # AphidCV Classifier YOLO settings
            results = model.predict(img, imgsz=128, verbose=False)
            # Foi treinado com imgsz=120, mas o valor deve ser múltiplo de 32.
            # Conferido no log de treino, é passado 120, mas a arquitetura redefine para 128.

            if(results[0].boxes):
                object_confidence = results[0].boxes.conf
                object_confidence = object_confidence.data[0].detach().item() # max(porcentagem[0])
                class_id = results[0].boxes.cls
                classe = int(class_id.data[0].detach().item())  # classe

                # 0: Winged
                # 1: Wingless
                # 2: False
                # 3: Nymph
                if(object_confidence < CONFIDENCE and classe != 2):
                    cv.imwrite(
                        'temp/' + 'img' + str(labels[classe]) + '-' + str(object_confidence) + '.jpg', img)

                leftmost = tuple(cnt[cnt[:, :, 0].argmin()][0])
                rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])
                topmost = tuple(cnt[cnt[:, :, 1].argmin()][0])
                bottommost = tuple(cnt[cnt[:, :, 1].argmax()][0])
                dist_x = (((rightmost[0] - leftmost[0])**2) +
                          ((rightmost[1] - leftmost[1])**2))**0.5
                dist_y = (((topmost[0] - bottommost[0])**2) +
                          ((topmost[1] - bottommost[1])**2))**0.5
                if(dist_x > dist_y):
                    length = dist_x/PXMM
                else:
                    length = dist_y/PXMM

                if(object_confidence < CONFIDENCE and (classe == 1 or classe == 3)):
                    # Verifica se é um áptero OU uma ninfa de ínstar alto (geralmente 4),
                    # em casos onde o valor de confiança é inferior a 80%
                    # Confere com o comprimento máximo da ninfa, definido por espécie, nas configurações do sistema
                    if(length > maxNymph):
                        classe = 1
                    else:
                        classe = 3

                if(object_confidence < CONFIDENCE and (classe == 0)):
                    # Verifica se é um alado, de fato, ou um elemento falso,
                    # em casos onde o valor de confiança é inferior a 80%
                    # Confere o intervalo entre o comprimento máximo de um afídeo e a
                    # diferença entre o comprimento máximo da ninfa e o comprimento mínimo de um afídeo
                    if(length > maxSize or length < (maxNymph-minSize)): # if(length > 3 or length < 1):
                        classe = 2
                    else:
                        classe = 0

                perimeter = cv.arcLength(cnt, True) / PXMM
                area = cv.contourArea(cnt)/(PXMM*PXMM)
                weight1 = 0.0002 * (area*area) + 0.0002 * area + 0.00005
                weight2 = 0.0006 * (length*length) + 0.0009 * length + 0.00005
                weight = (weight1 + weight2)/2.0
                totalWeight += weight

                # Define um valor de confiança FIXO, igual ou maior que 80%,
                # para confirmar contagem, classificação e mensuração nos arquivos de saída
                # Porém, mantém o desenho de tudo o que foi categorizado/ajustado na imagem de saída
                # No desenho, destaca aqueles que estão no intervalo de confiança com negrito e asterisco
                confidence = CONFIDENCE
                if classe == 0:
                    if(object_confidence >= confidence):
                        wingeds.append(result(nomeEspecie[especie], "Winged", str(
                            object_confidence), str(length), str(weight), str(perimeter), str(area)))
                        countWinged += 1
                    cv.drawContours(IMAGE, [cnt], 0, (255, 0, 255), 1)
                elif classe == 1:
                    if(object_confidence >= confidence):
                        wingless.append(result(nomeEspecie[especie], "Wingless", str(
                            object_confidence), str(length), str(weight), str(perimeter), str(area)))
                        countWingless += 1
                    cv.drawContours(IMAGE, [cnt], 0, (0, 0, 255), 1)
                elif classe == 2:
                    cv.drawContours(IMAGE, [cnt], 0, (255, 128, 0), 1)
                elif classe == 3:
                    if(object_confidence >= confidence):
                        nymphs.append(result(nomeEspecie[especie], "Nymph", str(
                            object_confidence), str(length), str(weight), str(perimeter), str(area)))
                        countNymphs += 1
                    cv.drawContours(IMAGE, [cnt], 0, (0, 255, 255), 1)

                cv.putText(IMAGE, "A: " + str('{:.2f}'.format(area)), (cx + 10, cy + 10),
                        cv.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 0), 1, cv.LINE_AA, False)
                cv.putText(IMAGE, "P: " + str('{:.2f}'.format(perimeter)), (cx + 10, cy + 20),
                        cv.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1, cv.LINE_AA, False)
                cv.putText(IMAGE, "L: " + str('{:.2f}'.format(length)), (cx + 10, cy + 30),
                        cv.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1, cv.LINE_AA, False)
                cv.putText(IMAGE, "W: " + str('{:.4f}'.format(weight)), (cx + 10, cy + 40),
                        cv.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 255), 1, cv.LINE_AA, False)
                if(object_confidence >= CONFIDENCE):
                    cv.putText(IMAGE, str('{:.2f}'.format(object_confidence*100)) + "% *", (
                        cx + 10, cy + 50), cv.FONT_HERSHEY_DUPLEX, 0.35, (0, 255, 255), 1, cv.LINE_AA, False)
                else:
                    cv.putText(IMAGE, str('{:.2f}'.format(object_confidence*100)) + "%", (
                        cx + 10, cy + 50), cv.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1, cv.LINE_AA, False)

    results = nymphs + wingless + wingeds
    countTotal = countWinged + countNymphs + countWingless

    cv.putText(IMAGE, "Total insects found: " + str(countTotal), (100, 100),
               cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4, cv.LINE_AA, False)
    cv.putText(IMAGE, "Total nymphs found: " + str(countNymphs), (100, 200),
               cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4, cv.LINE_AA, False)
    cv.putText(IMAGE, "Total wingless found: " + str(countWingless), (100, 300),
               cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4, cv.LINE_AA, False)
    cv.putText(IMAGE, "Total wingeds found: " + str(countWinged), (100, 400),
               cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4, cv.LINE_AA, False)
    cv.putText(IMAGE,  nomeEspecie[especie], (100, 500),
               cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4, cv.LINE_AA, False)
    cv.putText(IMAGE,  "Min Confidence: 0.80", (100, 600),
               cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4, cv.LINE_AA, False)
    cv.putText(IMAGE, "o", (60, 200),cv.FONT_HERSHEY_SIMPLEX, 2,
               (0, 255, 255), 4, cv.LINE_AA, False)
    cv.putText(IMAGE, "o", (60, 300), cv.FONT_HERSHEY_SIMPLEX, 2,
               (0, 0, 255), 4, cv.LINE_AA, False)
    cv.putText(IMAGE, "o", (60, 400),cv.FONT_HERSHEY_SIMPLEX, 2,
               (255, 0, 255), 4, cv.LINE_AA, False)

    # GERA EXCEL DETALHADO
    #==============================================================
    os.mkdir(os.path.join(folder_path, 'csv'))
    with open(os.path.join(folder_path, 'csv', 'resultado.csv'), 'w', newline='') as csvfile:
        filewriter = csv.writer(csvfile, dialect='excel-tab', delimiter=';')
        filewriter.writerow(["Specie", "Type", "Probability", "Length (mm)",
                             "Weight (mg)", "Perimeter (mm)", "Area (mm^2)"])
        for r in results:
            filewriter.writerow([r.specie, r.classtype, r.probability.replace(".", ","), r.length.replace(
                ".", ","), r.weight.replace(".", ","), r.perimeter.replace(".", ","), r.area.replace(".", ",")])
    if int(manual_Total) == 0:
        manual_Total = int(manual_Nymphs) + int(manual_Wingeds) + int(manual_Wingless)

    # GERA EXCEL GERAL
    with open(os.path.join(folder_path, 'Amostra Teste AphidCV.csv'), 'w', newline='') as csvfile:
        filewriter = csv.writer(csvfile, dialect='excel-tab', delimiter=';')
        filewriter.writerow(["Image Label", "QuantityNymphs",
                             "QuantityWingless", "QuantityWingeds", "Total", "Total Weight",
                             "ManualNymphs", "ManualWingless", "ManualWingeds", "ManualTotal",
                             "especie", "diameter", "thresh", "minsize", "maxsize", "maxnymph", "contrast", "brightness"])
        filewriter.writerow([FILE, countNymphs, countWingless,
                             countWinged, countTotal, str(totalWeight).replace(".", ","), manual_Nymphs, manual_Wingless, manual_Wingeds, manual_Total,
                             str(especie), str(diameter), str(thresh), str(minSize), str(maxSize), str(maxNymph), str(contrast), str(brightness)])

    # CRIAÇÃO DO JSON
    data = {"label": "Amostra Teste AphidCV", "responsible": "Embrapa Trigo", "collectionLatitude": "12412412.222", "collectionLongitude": "4124244.22",
            "processingDate": str(datetime.now()), "collectionDate": str(datetime.now()),
            "images": [{"name": FILE + ".jpg", "quantityNymphs": countNymphs, "quantityWinged": countWinged, "quantityWingless": countWingless, "quantityTotal": countTotal, "manualNymphs": manual_Nymphs, "manualWinged": manual_Wingeds, "manualWingless": manual_Wingless, "manualTotal": manual_Total, "weightTotal": totalWeight,
                        "nymph": [n.__dict__ for n in nymphs], "wingless":[wl.__dict__ for wl in wingless], "winged": [w.__dict__ for w in wingeds]}]}
    # GERA ARQUIVO JSON
    with open(os.path.join(folder_path, 'Amostra Teste AphidCV.json'), 'w') as outfile:
        outfile.write(json.dumps(data))

    # GERA IMAGEM ROTULADA
    cv.imwrite(os.path.join(folder_path, "resultado.jpg"), IMAGE)
    print(f'PXMM: {PXMM}')
    print(f'Total: {countTotal}')

    shutil.rmtree(os.path.join(folder_path, 'temp'), ignore_errors=True)

    return pasta

#=====================================================================================================

# CAMINHO= "/home/upf/Downloads/"
# FILE= CAMINHO + "3.jpeg"

# Dicionário de espécies
especies = {
    'rp': 'Rhopalosiphum padi',
    'sg': 'Schizaphis graminum',
    'md': 'Metopolophium dirhodum',
    'sa': 'Sitobion avenae',
    'mp': 'Myzus persicae',
    'bb': 'Brevicoryne brassicae',
    'rp_gen1v2': 'Rhopalosiphum padi GEN1v2',
    'sg_gen1v2': 'Schizaphis graminum GEN1v2',
    'sa_gen1v2': 'Sitobion avenae GEN1v2',
    'md_gen1v2': 'Metopolophium dirhodum GEN1v2'
}

# Dicionário de modelos para cada espécie
modelos = {
    'rp': 'modelos/rp.pt',
    'sg': 'modelos/sg.pt',
    'md': 'modelos/md.pt',
    'sa': 'modelos/sa.pt',
    'mp': 'modelos/mp.pt',
    'bb': 'modelos/bb.pt',
    'rp_gen1v2': 'modelos/rp_gen1v2.pt',
    'sg_gen1v2': 'modelos/sg_gen1v2.pt',
    'sa_gen1v2': 'modelos/sa_gen1v2.pt',
    'md_gen1v2': 'modelos/md_gen1v2.pt'
}

def main():
    # Cria o parser para entrada de parâmetros
    parser = argparse.ArgumentParser(description="Escolha uma imagem para realizar a detecção e defina os parâmetros:")
    
    # Adiciona o argumento para selecionar o arquivo
    parser.add_argument('arquivo', type=str)
    
    # Adiciona um menu suspenso para escolher a espécie
    parser.add_argument('--especie', type=str, choices=list(especies.keys()), default='rp')
    
    # Argumentos adicionais (ainda não usados)
    parser.add_argument('--contrast', type=float, default=1.01)
    parser.add_argument('--brightness', type=int, default=0)
    
    # Captura os argumentos do parser
    args = parser.parse_args()
    
    # Chama a função que processa a imagem
    process_image(args, modelos[args.especie])


def process_image(args, model_path):
    FILE = args.arquivo
    CAMINHO = os.path.dirname(FILE)
    especie = args.especie
    minSize = 0.3
    maxSize = 3.0
    maxNymph = 1.5
    thresh = 115
    diameter = 140
    contrast = args.contrast
    brightness = args.brightness
    manual_Nymphs = 0
    manual_Wingless = 0
    manual_Wingeds = 0
    manual_Total = 0
    
    # Exibe o caminho do modelo selecionado
    print("===========================")
    print(f"Modelo selecionado para a espécie '{especie}': {model_path}")
    print("===========================")
    
    # Carrega o modelo YOLO com base na espécie
    modelo_yolo = YOLO(model_path)
    
    image = clean_image(FILE, especie, 0, [0,0,0], thresh)
    pxmm = image[2] / (diameter/2)
    
    folder = count_object(image[0], image[1], pxmm, CAMINHO, FILE, especie, minSize, maxSize, maxNymph, thresh, diameter, contrast, brightness, manual_Nymphs, manual_Wingless, manual_Wingeds, manual_Total, modelo_yolo)


if __name__ == "__main__":
    main()


'''
    nomeEspecie = {
    'rp': 'Rhopalosiphum padi',
    'sg': 'Schizaphis graminum',
    'md': 'Metopolophium dirhodum',
    'sa': 'Sitobion avenae',
    'mp': 'Myzus persicae',
    'bb': 'Brevicoryne brassicae',
    'rp_gen1v2': 'Rhopalosiphum padi GEN1v2',
    'sg_gen1v2': 'Schizaphis graminum GEN1v2',
    'sa_gen1v2': 'Sitobion avenae GEN1v2',
    'md_gen1v2': 'Metopolophium dirhodum GEN1v2'}
'''

# -*- coding: utf-8 -*-
"""
MLP classificador de gestos
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
import glob
from sklearn.preprocessing import normalize
from sklearn.neural_network import MLPClassifier
import ntpath
import random

def getHuMoments(path):
    """
    Método responsável por pegar as imagens de um diretório e calcular os momentos de hu
    Retorno: Array de arrays dos momentos calculados [[m11,m12,m13,m14,m15,m16,m17], [m21,m22,m23,m24,m25,m26,m27] ...]
    """
    files = glob.glob(path)

    matrixImg = []
    for imgFile in files:
        imgReaded = cv2.imread(imgFile)
        imageGray = cv2.cvtColor(imgReaded, cv2.COLOR_BGR2GRAY)
        moments = cv2.HuMoments(cv2.moments(imageGray)).flatten()
        matrixImg.append(moments)
        
    matrixImgNormalized = normalize(matrixImg, norm='max', axis=0).tolist()
    return matrixImgNormalized
        
def getClasses(path):
    """
    Método responsável por pegar as imagens de um diretório e pega as classes de acordo com a nomenclatura dos arquivos.
    A nomenclatura do arquivo segue a seguinte regra: 1001_1_2.png => (1001 = prefixo, 1 = classe, 2 = segundo elemento do cojunto dessa classe)
    Retorno: Array de arrays dos momentos calculados [c1,c2,c3,c4,c5,c6,c7,c8,c9...]
    """
    files = glob.glob(path)
    classes = []
    for imgFile in files:
        #concatenando a class
        tempName = ntpath.basename(imgFile) # 1001_1_2.png => (1001 = prefixo, 1 = classe, 2 = segundo elemento do cojunto dessa classe)
        classes.append(int(tempName.split("_")[1])) # a classe da figura esta definida no 
    return classes
    
def showImageAndCalcHu(pathImg):
    img = cv2.imread(pathImg)
    imageGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    moments = cv2.HuMoments(cv2.moments(imageGray)).flatten()
    print moments
    
    plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
    plt.show()

#showImageAndCalcHu("C:\\Users\\user\\Desktop\\Classificador\\training3\\1001_5_970.png")
    
imgsHus = getHuMoments("C:\\Users\\user\\Desktop\\Classificador\\training3\\*.png")
classes = getClasses("C:\\Users\\user\\Desktop\\Classificador\\training3\\*.png")

#separando um conjunto aleatorio para teste
imgsHusPredict = []
classesPredict = []
for i in range(0,300):
    idx = random.randint(0,len(imgsHus)-1)
    imgsHusPredict.append(imgsHus[idx])
    classesPredict.append(classes[idx])
    imgsHus = np.delete(imgsHus, idx, axis=0).tolist()
    classes = np.delete(classes, idx).tolist()

#criando MLP
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(10,), random_state=1)
#treinando MLP
clf.fit(imgsHus, classes)
#testando MLP
aux = clf.predict(imgsHusPredict).tolist()

#calculando taxa de acerto
err = 0
for i in range (0,len(aux)):
    if aux[i] != classesPredict[i]:
        err+= 1
    print 'Predicted: ' + str(aux[i]) + ' | Real Class: ' + str(classesPredict[i])
    
print 'Numero de erros: ' + str(err)
print '% Err: ' + str(err/float(len(aux)))
    

# Reconhecimento facial em Python com a biblioteca OpenCV
# Este codigo nao utiliza leitura de webcam, apenas imagens

import cv2 # importando biblioteca OpenCV

# Download XML Modelo
# https://github.com/opencv/opencv/tree/master/data/haarcascades

# Download Biblioteca OpenCV
# https://www.lfd.uci.edu/~gohlke/pythonlibs/
# 
# Como instalar?
# 1 - Faça o download
# 2 - Execute CMD como ADMIN
# 3 - Execute o comando CD caminho do download
# Exemplo: CD C:\Users\python\Downloads
# 4 - Execute o comando py -m pip install + arquivo de download
# Exemplo: py -m pip install opencv_python_headless-4.5.2+dummy-py3-none-any.whl
# 5 - Ok será instalado a biblioteca
#

image_path = 'foto.jpg' #nome do arquivo de origem
cascade_path = 'haarcascade_frontalface_default.xml' #arquivo de modelo XML

clf = cv2.CascadeClassifier(cv2.data.haarcascades + cascade_path)

img = cv2.imread(image_path) #parametro ler imagem

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = clf.detectMultiScale(gray, 1.3, 10)

for (x, y, w, h) in faces: #ler rostos nas imagens
	img = cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 0), 2) 

cv2.imshow('Visualizador', img)
cv2.waitKey(0)
cv2.destroyALlWindows ()

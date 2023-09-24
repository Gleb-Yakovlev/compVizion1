import cv2
import os

from tkinter import *
from tkinter import ttk

from matplotlib import pyplot as plt
import numpy as np

st = os.path.join(os.path.dirname(__file__),'standart.jpeg')
assert os.path.exists(st)#проверка на существование файла
standart = cv2.imread(st)

fr = os.path.join(os.path.dirname(__file__),'fragment.jpg')
assert os.path.exists(fr)#проверка на существование файла
fragment = cv2.imread(fr)

st = os.path.join(os.path.dirname(__file__),'Statham.jpg')
assert os.path.exists(st)#проверка на существование файла
statham = cv2.imread(st)

#show standart image
def showImage():
    cv2.imshow('standart', standart)
    cv2.waitKey(0)

#show fragment image
def showFragment():
    cv2.imshow('fragment', fragment)
    cv2.waitKey(0)

#inline function match template
def inMatchTemplate():
    res  = cv2.matchTemplate(standart, fragment, cv2.TM_CCOEFF_NORMED)
    cv2.namedWindow('res', cv2.WINDOW_KEEPRATIO)
    cv2.imshow('res', res)
    cv2.resizeWindow('res', 400, 400)

    plt.subplot(121), plt.imshow(standart, cmap='gray'),
    plt.title('Standart'), plt.axis('off')
    plt.subplot(122), plt.imshow(fragment, cmap='gray'),
    plt.title('Fragment'), plt.axis('off')
    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#match template
def matchTemplate():
    res = np.zeros((1600, 1600))
    count = 0
    H1, W1 = standart.shape[:2]
    print(H1, W1)
    for Fy in range(H1 - 100):
        print("Fy ", Fy)
        for Fx in range(W1 - 100):
            #print("Fx ", Fx)
            cropStand = standart[Fy:Fy+100, Fx:Fx+100]
            # переводим изображения в оттенки серого
            grayS = cv2.cvtColor(cropStand, cv2.COLOR_BGR2GRAY)
            grayF = cv2.cvtColor(fragment, cv2.COLOR_BGR2GRAY)
            # вычисляем размеры изображений
            h1, w1 = grayS.shape[:2]
            h2, w2 = grayF.shape[:2]
            # вычисляем средние значения яркости для каждого изображения
            mean1 = cv2.mean(grayS)[0]
            mean2 = cv2.mean(grayF)[0]
            # вычисляем дисперсии яркости для каждого изображения
            var1 = np.sum((grayS - mean1) ** 2) / (h1 * w1)
            var2 = np.sum((grayF - mean2) ** 2) / (h2 * w2)
            # вычисляем ковариацию между изображениями
            cov = np.sum((grayS - mean1) * (grayF - mean2)) / (h1 * w1)
            # вычисляем корреляционную функцию
            corr = cov / np.sqrt(var1 * var2)
            res[Fx][Fy] = corr
            #print("Correlation:", corr)
            count += 1
    print(count)
    cv2.namedWindow('result', cv2.WINDOW_KEEPRATIO)
    cv2.imshow('result', res)
    cv2.resizeWindow('result', 400, 400)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#turn
def turnDef():
    rot = -10
    count = 1
    while(rot != 12):
        (h, w) = standart.shape[:2]
        center = (int(w / 2), int(h / 2))
        rotation_matrix = cv2.getRotationMatrix2D(center, rot, 1)
        rotatedStandart = cv2.warpAffine(standart, rotation_matrix, (w, h))
        res = cv2.matchTemplate(rotatedStandart, fragment, cv2.TM_CCOEFF_NORMED)
        cv2.namedWindow('custom window' + str(rot), cv2.WINDOW_KEEPRATIO)
        cv2.imshow('custom window' + str(rot), res)
        cv2.resizeWindow('custom window' + str(rot), 400, 400)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # plt.subplot(2, 6, count)
        # plt.title(rot), plt.imshow(res, cmap='gray')
        # plt.axis('off')
        rot = rot + 2
        count += 1
    plt.show()
#find Statham
def findStatham():
    res  = cv2.matchTemplate(standart, statham, cv2.TM_CCOEFF_NORMED)
    cv2.namedWindow('res', cv2.WINDOW_KEEPRATIO)
    cv2.imshow('res', res)
    cv2.resizeWindow('res', 400, 400)
    plt.subplot(131), plt.imshow(res, cmap='gray'),
    plt.title('Matching Result'), plt.axis('off')
    plt.subplot(121), plt.imshow(standart, cmap='gray'),
    plt.title('Detected Point'), plt.axis('off')
    plt.subplot(122), plt.imshow(statham, cmap='gray'),
    plt.title('Statham'), plt.axis('off')
    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
#scaling
def scaling():
    scaleKoef = 0.9
    step = 0.025
    count = 1
    while(scaleKoef <= 1.1):
        scaleStandart = standart
        width = int(scaleStandart.shape[1]*scaleKoef)
        height = int(scaleStandart.shape[0]*scaleKoef)
        dsize = (width, height)
        scaleStandart = cv2.resize(standart, dsize)
        res = cv2.matchTemplate(scaleStandart, fragment, cv2.TM_CCOEFF_NORMED)
        cv2.namedWindow('custom window' + str(scaleKoef), cv2.WINDOW_KEEPRATIO)
        cv2.imshow('custom window' + str(scaleKoef), res)
        cv2.resizeWindow('custom window' + str(scaleKoef), 400, 400)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        scaleKoef += step
        count += 1
        print(count)
    plt.show()
    print("done")




#-------menu-------
root = Tk()
root.title("Menu")
root.geometry("250x250")

btn1 = ttk.Button(text="Show standart image", command=showImage)
btn1.pack()
btn2 = ttk.Button(text="Show fragment image", command=showFragment)
btn2.pack()
btn3 = ttk.Button(text="Inline function", command=inMatchTemplate)
btn3.pack()
btn4 = ttk.Button(text="Match template (dont click!)", command=matchTemplate)
btn4.pack()
btn5 = ttk.Button(text="Rotation match template", command=turnDef)
btn5.pack()
btn6 = ttk.Button(text="Try find Statham", command=findStatham)
btn6.pack()
btn7 = ttk.Button(text="Scaling match template", command=scaling)
btn7.pack()

root.mainloop()
#-------menu-------

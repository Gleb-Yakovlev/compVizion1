import cv2
import os

from tkinter import *
from tkinter import ttk

from matplotlib import pyplot as plt

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
    #cv2.imshow('fragment', fragment)
    #cv2.waitKey(0)
    res  = cv2.matchTemplate(standart, fragment, cv2.TM_CCOEFF)
    #min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    plt.subplot(131), plt.imshow(res, cmap='gray'),
    plt.title('Matching Result'), plt.axis('off')
    plt.subplot(132), plt.imshow(standart, cmap='gray'),
    plt.title('Standart'), plt.axis('off')
    plt.subplot(133), plt.imshow(fragment, cmap='gray'),
    plt.title('Fragment'), plt.axis('off')
    plt.show()
#match template
def matchTemplate():
    cv2.imshow('fragment', fragment)
#turn
def turnDef():
    (h, w) = standart.shape[:2]
    center = (int(w / 2), int(h / 2))
    rotation_matrix = cv2.getRotationMatrix2D(center, -6, 1)
    rotatedStandart = cv2.warpAffine(standart, rotation_matrix, (w, h))

    (h, w) = fragment.shape[:2]
    center = (int(w / 2), int(h / 2))
    rotation_matrix = cv2.getRotationMatrix2D(center, -6, 1)
    rotatedFragment = cv2.warpAffine(fragment, rotation_matrix, (w, h))
    
    res = cv2.matchTemplate(rotatedStandart, rotatedFragment, cv2.TM_CCOEFF)
    plt.subplot(131), plt.imshow(res, cmap='gray'),
    plt.title('Matching Result'), plt.axis('off')
    plt.subplot(132), plt.imshow(rotatedStandart, cmap='gray'),
    plt.title('Standart'), plt.axis('off')
    plt.subplot(133), plt.imshow(rotatedFragment, cmap='gray'),
    plt.title('Fragment'), plt.axis('off')
    plt.show()
#find Statham
def findStatham():
    res  = cv2.matchTemplate(standart, statham, cv2.TM_CCOEFF)
    #min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    plt.subplot(131), plt.imshow(res, cmap='gray'),
    plt.title('Matching Result'), plt.axis('off')
    plt.subplot(132), plt.imshow(standart, cmap='gray'),
    plt.title('Detected Point'), plt.axis('off')
    plt.subplot(133), plt.imshow(statham, cmap='gray'),
    plt.title('Statham'), plt.axis('off')
    plt.show()




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
btn4 = ttk.Button(text="Match template", command=matchTemplate)
btn4.pack()
btn5 = ttk.Button(text="Rotation match template", command=turnDef)
btn5.pack()
btn6 = ttk.Button(text="Try find Statham", command=findStatham)
btn6.pack()

root.mainloop()
#-------menu-------

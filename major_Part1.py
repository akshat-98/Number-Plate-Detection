# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 13:43:23 2019

@author: Naman Kalra
"""
import numpy as np
import cv2
import  imutils
import sys
import pytesseract
import pandas as pd
import time

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

filename = './7.mp4'

cap = cv2.VideoCapture(filename)
count = 0
while cap.isOpened():
    ret,frame = cap.read()
    if ret == True:
        cv2.imshow('window-name',frame)
        cv2.imwrite("./output/frame%d.jpg" % count, frame)
        count = count + 1
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()

image = cv2.imread("./output/frame%d.jpg"%(count-1))

for angle in np.arange(0, 105, 15):
    rotated = imutils.rotate_bound(image, angle)
    cv2.imwrite("rotatedimage.jpg", rotated)

image = imutils.resize(rotated, width=500)
cv2.imwrite("resized.jpg", image)

#Plate Localization
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imwrite("Gray.jpg", gray)
cv2.imshow('Gray', gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

gray = cv2.bilateralFilter(gray, 11, 17, 17)

edged = cv2.Canny(gray, 170, 200)
cv2.imwrite("Edged.jpg", edged)
cv2.imshow('Edged', edged)
cv2.waitKey(0)
cv2.destroyAllWindows()
​
(cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

cnts=sorted(cnts, key = cv2.contourArea, reverse = True)[:20]
NumberPlateCnt = None 

count = 0
for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:  
            NumberPlateCnt = approx 
            break
​
mask = np.zeros(gray.shape,np.uint8)
new_image = cv2.drawContours(mask,[NumberPlateCnt], 0, 55, -1)
new_image = cv2.bitwise_and(image, image, mask=mask)

cv2.imwrite("Final_image.jpg", new_image)
cv2.imshow('Final', new_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


#Character Segmentation
(x, y) = np.where(mask == 55)
(topx, topy) = (np.min(x), np.min(y))
(bottomx, bottomy) = (np.max(x), np.max(y))
Cropped = gray[topx:bottomx+1, topy:bottomy+1]

cv2.imwrite("Cropped.jpg", Cropped)
cv2.imshow('NoPlate', Cropped)
cv2.waitKey(0)
cv2.destroyAllWindows()


#Character Recognition
config = ('-l eng --oem 1 --psm 3')
text = pytesseract.image_to_string(Cropped, config=config)
raw_data = {'date':[time.asctime( time.localtime(time.time()))],'No plate':[text]}

df = pd.DataFrame(raw_data)
df.to_csv('data.csv',mode='a')
​
print("Detected Number is: ",text)
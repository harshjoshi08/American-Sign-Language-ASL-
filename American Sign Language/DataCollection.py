import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(0) #0 for default camera
detector = HandDetector(detectionCon=0.8, maxHands=1)
offset = 20
imgSize = 300
counter = 0 

# folder = "C:\Harsh Works\code\American Sign Language\Safe Side\Data\Hello"
# folder = "C:\Harsh Works\code\American Sign Language\Safe Side\Data\Thank you"
folder = "C:\Harsh Works\code\American Sign Language\Safe Side\Data\Yes"
# folder = "C:\Harsh Works\code\American Sign Language\Safe Side\Data\I Love You"
# folder = "C:\\Harsh Works\\code\\American Sign Language\\Data\\no"
# folder = "C:\Harsh Works\code\American Sign Language\Safe Side\Data\Okay"
# folder = "C:\Harsh Works\code\American Sign Language\Safe Side\Data\Please"

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imgwhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255 # White background
        imgcrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
        imgcropshape = imgcrop.shape
        aspectratio = h / w
        
        if aspectratio > 1:
            k = imgSize / h
            wcal = math.ceil(k * w)
            imgresize = cv2.resize(imgcrop, (wcal, imgSize))
            imgresizeShape = imgresize.shape
            wgap = math.ceil((imgSize - wcal) / 2)
            imgwhite[:, wgap:wcal + wgap] = imgresize
        
        else:
            k = imgSize / w
            hcal = math.ceil(k * h)
            imgresize = cv2.resize(imgcrop, (imgSize, hcal))
            imgresizeShape = imgresize.shape
            hgap = math.ceil((imgSize - hcal) / 2)
            imgwhite[hgap:hcal + hgap, :] = imgresize
            
        cv2.imshow("ImageCrop", imgcrop)
        cv2.imshow("ImageWhite", imgwhite)
        
    cv2.imshow("Image", img) 
    key = cv2.waitKey(1) # key meeans keyboard input
    
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgwhite)
        print(counter)       

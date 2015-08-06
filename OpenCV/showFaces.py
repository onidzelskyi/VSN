#!/usr/bin/python


import numpy as np
import cv2


image = "in.jpg"
face_cascade = cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
img = cv2.imread(image)
img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
  face = img[y:y+h,x:x+w]
  cv2.imwrite("face11.jpg",face)
  cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),4)
  cv2.imwrite("face10.jpg",img)
  #cv2.imshow("img",img)
  #cv2.waitKey(0)
  #cv2.destroyAllWindows()
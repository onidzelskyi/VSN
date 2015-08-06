#!/usr/bin/python


import numpy as np
import cv2


image = "/Users/alex/Documents/VSN/src/examples/clustering/person_clustering/data/1/0e8c084286a616c8432c165df9afc458"


face_detector = [
  cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml'),
  cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml'),
  cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt2.xml'),
  cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt_tree.xml'),
  cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_profileface.xml')
]


eye_detector = [
  cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_eye.xml'),
  cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_eye_tree_eyeglasses.xml'),
  cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_lefteye_2splits.xml'),
  cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_mcs_eyepair_big.xml'),
  cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_mcs_eyepair_small.xml'),
  cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_mcs_righteye.xml'),
  cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_righteye_2splits.xml'),
  cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_mcs_lefteye.xml')
]


mouth_detector = [
  cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_mcs_mouth.xml')
]


nose_detector = [
  cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_mcs_nose.xml')
]


A = {}
index = 1.0


img = cv2.imread(image)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
for fd in face_detector:
  faces = fd.detectMultiScale(gray, 1.3, 5)
  index = index - 0.25
  for (x,y,w,h) in faces:
    if x not in A and \
      (x-1) not in A and \
      (x-2) not in A and \
      (x+1) not in A and \
      (x+2) not in A:
      A[x] = x
      print x
      r = index * 255
      g = index * 255
      b = index * 255
      cv2.rectangle(img,(x,y),(x+w,y+h),(b,g,r),4)
      #eyes
      for ed in eye_detector:
        eyes = ed.detectMultiScale(gray[y:y+h, x:x+w])
        for (ex,ey,ew,eh) in eyes:
          cv2.rectangle(img[y:y+h, x:x+w],(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
      #mouth
      for md in mouth_detector:
        mouth = md.detectMultiScale(gray[y:y+h, x:x+w])
        for (ex,ey,ew,eh) in mouth:
          cv2.rectangle(img[y:y+h, x:x+w],(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
  print "___"
cv2.imwrite("test.jpg",img)

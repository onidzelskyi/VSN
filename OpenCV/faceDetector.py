# prerequisities
#sudo apt-get install python-opencv

import numpy as np
import cv2
import sys
import os
import hashlib
import logging
from collections import namedtuple
import time


OFFSET_X = 20
OFFSET_Y = 20
OFFSET_H = 20
OFFSET_W = 20

X = []
Y = []
K = 0

Entity = namedtuple("Entity", "dir, entity")


logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(3)

class TrainingDataParser(object):


  def __init__(self):
    self.Entities = []
    self.entity = None
    self.X = []
    self.Y = []
    self.training_data_dir = None
    self.gray = None
    self.face_cascade = cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
    #self.eye_cascade = cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_eye.xml')
    self.eye_cascade = cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_eye_tree_eyeglasses.xml')
  

  def readTrainingDataDir(self, training_data_dir):
    flag = True
    self.training_data_dir = training_data_dir
    try:
      self.entities = [self.generateEntity(dir) for dir in os.listdir(training_data_dir)]
      logger.info("Entities: %s", str(self.entities))
    except OSError:
      flag = False
      print training_data_dir + " not exist."
    return flag


  def generateEntity(self, dir):
    entity = str(hashlib.md5(dir).hexdigest())    
    logger.info("dir: <<%s>>", dir)
    logger.info("entity: <<%s>>", entity)
    return Entity(dir, entity)


  def parseEntities(self):
    flag = True
    for entity in self.entities:
      self.entity = entity
      self.parseEntity()
    #logger.info("X: %s", str(self.X))
    #logger.info("Y: %s", str(self.Y))
    return flag


  def parseEntity(self):
    flag = True
    logger.info("Parse entity: <<%s>>", str(self.entity))
    for file in os.listdir(self.training_data_dir + "/" + self.entity.dir):
      self.in_path = self.training_data_dir + "/" + self.entity.dir
      self.out_path = self.training_data_dir + "_out/" + self.entity.dir
      try:
        os.makedirs(self.out_path)
      except OSError: pass
      self.parseFileFromEntity(self.in_path + "/" + file)
    return flag


  def generateFileName(self, file, good):
    time.sleep(1)
    prefix = "/"
    if not good:
        prefix = "/BAD_"
    return self.out_path + prefix + str(hashlib.md5(str(time.time())).hexdigest()) + ".png"


  def checkFace(self, candidate,x,y,w,h):
    flag = True
    roi_gray = self.gray[y:y+h, x:x+w]
    eyes = self.eye_cascade.detectMultiScale(roi_gray)
    logger.info("Eyes: %s" % str(len(eyes)))
    #faces = self.face_cascade.detectMultiScale(candidate, 1.3, 5)
    if len(eyes)==0:
      flag = False
    return flag


  def adjustCandidateArea(self, x,y,w,h):
    ax,ay,aw,ah = x,y,w,h
    if (x - OFFSET_X) > 0:
      ax = x - OFFSET_X
    else:
      ax = 0
    if (y - OFFSET_Y) > 0:
      ay = y - OFFSET_Y
    else:
      ay = 0
    if (x + w + OFFSET_W) < (self.image_width):
      aw = w + OFFSET_W + (x-ax)
    else:
      aw = self.image_width - ax
    if (y + h + OFFSET_H) < (self.image_height):
      ah = h + OFFSET_H + (y-ay)
    else:
      ah = self.image_height - ay
    return (ax,ay,aw,ah)

    
  def parseFileFromEntity(self, file, multi = False):
    global K
    flag = True
    candidates = 0
    logger.info("file: <<%s>>", file)    
    try:
      img = cv2.imread(file)
      self.image_height, self.image_width, self.image_depth = img.shape
      self.gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      faces = self.face_cascade.detectMultiScale(self.gray, 1.3, 5)
      candidates = len(faces)
      # Number of clusters
      K = max([K, candidates])
      logger.info("file: <<%s>> contain %s candidates" % (file, str(candidates)))
      logger.info("Number of clusters: %s" % str(K))
      if candidates>1:
        logger.info("More then 1 candidate - skip image")
        #flag = False
      if flag:
        for (x,y,w,h) in faces:
          logger.info("Before: x: %s, y: %s, w: %s, h: %s" % (x,y,w,h))
          (ax,ay,aw,ah) = self.adjustCandidateArea(x,y,w,h)
          logger.info("After: x: %s, y: %s, w: %s, h: %s" % (ax,ay,aw,ah))
          flag = self.checkFace(img[ay:ay+ah, ax:ax+aw],ax,ay,aw,ah)
          if flag:
            candidates = candidates + 1
            self.Y.append(self.entity.entity)
            self.X.append(img[y:y+h, x:x+w])
          file_name = self.generateFileName(file, good = flag)
          logger.info("Valid candidate: file_name: <<%s>>", file_name)
          cv2.imwrite(file_name,img[y:y+h, x:x+w])
    except Exception, err:
      logger.error(err.message)
      logger.error("File: <<%s>> is not valid image.", file)
      flag = False
    return flag


if __name__ == "__main__":
  trainingDataParser = TrainingDataParser()
  if trainingDataParser.readTrainingDataDir(sys.argv[1]):
    trainingDataParser.parseEntities()
  else:
    sys.exit(1)
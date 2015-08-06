#!/usr/bin/python


from pybrain.tools.shortcuts import buildNetwork
from pybrain.tools.validation import CrossValidator
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import TanhLayer
from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer, BiasUnit
from pybrain.structure import FullConnection
import operator
import sys
import os
import getopt
import argparse
import cv2
import numpy


X = []
Y = []
y = []


def loadWeights():
  weights = []
  with open("ThetaFull.csv","rb") as f:
    lines = f.readlines()
    for line in lines:
      weights = [float(a) for a in line.strip().split(',')]
  return weights


def loadDataSet(ds_file):
  global X, Y
  BB = set()
  aaa = {}
  ds = SupervisedDataSet(400, 10)
  #ds = SupervisedDataSet(1024, 5)
  with open(ds_file,"rb") as f:
    lines = f.readlines()
    for line in lines:
      l = [float(a) for a in line.strip().split(',')]
      #A = [float(1.0)] + l[:-1]
      A = l[:-1]
      X.append(A)
      B = int(l[-1])
      #BB.update([B])
      #for aa,bb in enumerate(BB):
      #  aaa[bb] = aa
      #print aaa
      #Y.append(aaa[bb])
      Y.append(B)
      C = []
      for i in range(10):
        C.append(int(1) if i==B or (i==0 and B==10) else int(0))
      ds.addSample(tuple(A), tuple(C))
  return ds


def LoadNN():
  #"""
  n = buildNetwork(1024, 512, 5)
  return n
  #"""
  n = FeedForwardNetwork()
  bias = BiasUnit()
  bias2 = BiasUnit()
  inLayer = LinearLayer(1024)
  hiddenLayer = SigmoidLayer(512)
  outLayer = LinearLayer(5)
  n.addInputModule(inLayer)
  n.addModule(hiddenLayer)
  n.addModule(bias)
  n.addModule(bias2)
  n.addOutputModule(outLayer)
  in_to_hidden = FullConnection(inLayer, hiddenLayer)
  hidden_to_out = FullConnection(hiddenLayer, outLayer)
  n.addConnection(in_to_hidden)
  n.addConnection(hidden_to_out)
  n.addConnection(FullConnection(bias, hiddenLayer))
  n.addConnection(FullConnection(bias2, outLayer))
  n.sortModules()
  return n


def checkAccuracy(net, trainer, ds):
  global y
  for x in X:
    a = net.activate(x)
    yy, value = max(enumerate(a), key=operator.itemgetter(1))
    y.append(yy+1)
    #y.append(a)
  index = 0
  for i in range(len(Y)):
    index = index + 1 if Y[i]==y[i] else index
    print Y[i], y[i]
  print float(index)/float(len(Y))
  #cv = CrossValidator(trainer, ds)
  #print cv.validate()


def readCMD():
  parser = argparse.ArgumentParser()
  parser.add_argument('-t', dest='train_data', action='store_true')
  parser.add_argument('-d', '--data')
  args = parser.parse_args()
  # create train data csv file
  if args.train_data and args.data is not None:
    createTrainData(args.data)


def createTrainData(data):
  A = None#numpy.array([])
  for dirname, dirnames, filenames in os.walk(data):
    if len(filenames):
      for file in filenames:
        if file[0]!=".":
          pid = int(dirname.split("/")[1])
          print pid, file
          image_name = dirname + "/" + file
          img = cv2.imread(image_name)
          img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
          img = img/255.
          image_height, image_width= img.shape
          min_side = min([image_width, image_height])
          x = int((image_width-min_side)/2.)
          y = int((image_height-min_side)/2.)
          reshaped_image = img[y:y+min_side, x:x+min_side]
          img = reshaped_image
          image_height, image_width = img.shape
          img = cv2.resize(img,(32,32))
          image_height, image_width = img.shape
          A = numpy.hstack((numpy.reshape(img, -1),pid)) if A is None else numpy.vstack((A, numpy.hstack((numpy.reshape(img, -1),pid))))
  numpy.savetxt("foo.csv", A, fmt="%s", delimiter=",")


def createClassifier(label):
  net = buildNetwork(400, 1)
  ds = SupervisedDataSet(400, 1)
  with open(ds_file,"rb") as f:
    lines = f.readlines()
    for line in lines:
      l = [float(a) for a in line.strip().split(',')]
      A = l[:-1]
      B = int(l[-1])
      C = [B] if B==label else [0]
      ds.addSample(tuple(A), tuple(C))
  trainer = BackpropTrainer(net, ds)
  trainer.train()
  return net

def oneVSAll():
  num_labels = 10
  nets = {}
  for label in range(num_labels):
    nets[label+1] = createClassifier(label+1)
  index = 0
  for a in range(5000):
    AA = []
    for label in range(num_labels):
      AA.append(nets[label+1].activate(X[a]))
    yy, value = max(enumerate(AA), key=operator.itemgetter(1))
    index = index+1 if Y[a]==(yy+1) else index
  print float(index)/len(Y)

if __name__=="__main__":
  #readCMD()
  #os._exit(0)
  ds_file = "./digits/data_full.csv"
  #ds_file = "foo.csv"
  ds = loadDataSet(ds_file)
  oneVSAll()
  os._exit(0)
  #w = loadWeights()
  net = LoadNN()
  #net._setParameters(w)
  #print net.params, len(net.params)
  trainer = BackpropTrainer(net, ds)
  trainer.train()
  #print net.params, len(net.params)
  checkAccuracy(net, trainer, ds)

#!/usr/bin/python


import sys
import os
import numpy
import cv2
import operator


WIDTH = 32
HEIGHT = 32


if __name__=="__main__":
    exit_code = 0
#try:
    if len(sys.argv) != 2:
      print "./scaleImages.py dirname"
      raise
    # get image need to be scaled catalog dir name
    dir = sys.argv[1]
    # truncate ending '/' symbol if exists
    dir = dir[:-1] if dir[-1]=="/" else dir
    # create output dir name
    out_dir = dir + "_scaled/"
    # create output directory
    try:
      os.makedirs(out_dir)
    except OSError:
      print "Cannot create output catalog: ", out_dir
      raise
    A = {}
    for dirname, dirnames, filenames in os.walk(dir):
      if len(filenames):
        for file in filenames:
          if file[0]!=".":
            #print dirname, "/", file
            image_name = dirname + "/" + file
            scaled_image_name = out_dir + "/" + dirname + "/" + file
            img = cv2.imread(image_name)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            image_height, image_width= img.shape
            print "original size: ", image_height, image_width
            #crop image
            # find min side size
            min_side = min([image_width, image_height])
            print "minimal side size: ", min_side
            #reshape image
            x = int((image_width-min_side)/2.)
            y = int((image_height-min_side)/2.)
            print "new points: ", x, y, y+min_side, x+min_side
            reshaped_image = img[y:y+min_side, x:x+min_side]
            img = reshaped_image
            #show image
            """
            cv2.imshow("img", reshaped_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            print "A"
            continue
            """
            #A[image_name] = { "image_height":image_height, "image_width":image_width, "image_depth":image_depth}
            image_height, image_width = img.shape
            print "reshaped size: ", image_height, image_width
            A[image_name] = { "image_height":image_height, "image_width":image_width}
            img = cv2.resize(img,(WIDTH,HEIGHT))
            image_height, image_width = img.shape
            print "resized size: ", image_height, image_width
            print scaled_image_name
            try:
              os.makedirs(out_dir + "/" + dirname)
            except OSError: pass
            cv2.imwrite(scaled_image_name,img)
#print A
#    print sorted(A.items(), key=operator.itemgetter(1))
    #except Exception, err:
#print err.message
#    exit_code = 1
    os._exit(exit_code)
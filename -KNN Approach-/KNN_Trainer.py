import sys
sys.path.append('../_Core Functions_')
import Extractor
import os
import numpy as np
from PIL import Image


def train_images():
    FOLDER_NAME = "/-Averaged Approach-"
    
    os.chdir('..')
    root = os.getcwd()
    for x in range(10):
        os.chdir(root +"/train_images_sorted/" + str(x))
        for filename in os.listdir(os.getcwd()):
            read_image(filename)
            
                        
def read_image(filename):
    image = Extractor.getImage(filename)
    grayscale = Extractor.ImageToMatrix(image)
    r = np.zeros((grayscale.shape[0], grayscale.shape[1]), dtype=int)
    
    lr = []
    start = True
    
    for row in range(len(grayscale)):
        left = 0
        right = 0
        for col in range(len(grayscale[row])):
            if start and grayscale[row][col] != 255:
                start = False
                left = col
            elif not start and grayscale[row][col] == 255:
                start = True
                right = col-1
                lr.append(("Line "+str(row)+":",left,right))
        start = True
    return lr

#def record_left_right():

#def record_segements():

#def record_inc_dec():

if __name__ == "__main__":
    os.chdir('..')
    root = os.getcwd()
    os.chdir(root +"/train_images_sorted/0")
    print(read_image("00002.tif"))

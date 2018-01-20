import sys
sys.path.append('../_Core Functions_')
import Extractor
import os
import numpy as np
from PIL import Image

class LLN:
    def __init__(line):
        self.line = line
        self.connections = []
    def link_to(node):
        self.coonections.append(node)
    def __str__():
        return int(self.line[1]-self.line[0])

class LL:
    def __init__(self, head = None):
        self.head = head
    

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
        num = 0
        lst = []
        for col in range(len(grayscale[row])):
            if start and grayscale[row][col] != 255:
                start = False
                left = col
            elif not start and grayscale[row][col] == 255:
                start = True
                num += 1
                right = col-1
                lst.append(left)
                lst.append(right)
        if num != 0:
            lr.append(lst)
        start = True
    return lr

def record_left_right(lr):
    left = 0
    right = 0
    total = 0
    for i in range(len(lr)-1):
        seg = min(len(lr[i]),len(lr[i+1]))//2
        
        #print(seg)
        for j in range(seg):
            seg = seg -1
            if lr[i][0+2*seg] > lr[i+1][0+2*seg]:
                left += 1
            if lr[i][1+2*seg] < lr[i+1][1+2*seg]:
                right += 1
            total += 1
    return (left,right,total)

#def record_segements():

#def record_inc_dec():

if __name__ == "__main__":
    os.chdir('..')
    root = os.getcwd()
    os.chdir(root +"/train_images_sorted/0")
    lr = read_image("00002.tif")
    print(lr)
    print(record_left_right(lr))

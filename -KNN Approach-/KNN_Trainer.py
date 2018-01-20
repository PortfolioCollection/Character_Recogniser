import matplotlib.pyplot as plt
import sys
sys.path.append('../_Core Functions_')
import Extractor
import os
import numpy as np
from PIL import Image

def train_images():

    os.chdir('..')
    root = os.getcwd()
    os.chdir(root +"/train_images_sorted/" + "0")
    points = []
    for filename in os.listdir(os.getcwd()):
        lr = read_image(filename)
        points.append(record_left_right(lr))
    plot(points)
                        
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
        
        # print(seg)
        for j in range(seg):
            seg = seg -1
            if lr[i][0+2*seg] > lr[i+1][0+2*seg]:
                left += 1
            if lr[i][1+2*seg] < lr[i+1][1+2*seg]:
                right += 1
            total += 1
            
    # return (left,right,total)
    return (left/total, right/total)
def plot(data):
    x = []
    y = []
    for point in data:
        x.append(point[0])
        y.append(point[1])
    t = np.arange(0, 1, 0.2)
    plt.plot(t, t, 'g--', x, y, 'ro', markersize=1)
    plt.axis([0, 1, 0, 1])
    plt.show()

#def record_segements():

#def record_inc_dec():

if __name__ == "__main__":
    train_images()
    # print(lr)
    # print(record_left_right(lr))

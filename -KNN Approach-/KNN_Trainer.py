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
    points = []
    numbers=[]
    for number in range(10):
        os.chdir(root +"/train_images_sorted/" + str(number))
        for filename in os.listdir(os.getcwd()):
            image = Extractor.getImage(filename)
            lr = read_image(image)
            lean = record_left_right(lr)
            segments = record_segment(lr)
            points.append((lean,segments))
            numbers.append(number)
        print("Done "+str(number))
    #plot(points)
    os.chdir(root+"/-KNN Approach-")
    save_file(points,numbers, "save.txt")
                        
def read_image(image):
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
            
    #return (left,right,total)
    return left/total+right/total

def record_segment(lr):
    one = 0
    two = 0
    total = len(lr)-1
    for i in range(len(lr)-1):
        seg = len(lr[i])
        if seg == 2:
            one+=1
        elif seg == 4:
            two+=1
    #return (one, two, total)
    return one/(3*total) +  2*two/(3*total)
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

def save_file(array, numbers, filename):
    file=open(filename,"w")
    line = 1
    for element in array:
        file.write("line "+str(line)+": lean("+str(element[0])+") segment("+str(element[1])+") class("+str(numbers[line-1])+")\n")
        line+=1
    file.close()
#def record_segements():

#def record_inc_dec():

if __name__ == "__main__":
    train_images()
    # print(lr)
    #os.chdir('..')
    #root = os.getcwd()
    #os.chdir(root +"/train_images_sorted/" + "1")
    #lr = read_image("00004.tif")
    #print(record_segment(lr))

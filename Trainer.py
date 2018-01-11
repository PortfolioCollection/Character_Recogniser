import Extractor
import os
import numpy as np
from PIL import Image


def loop_images():
    for x in range(10):
        os.chdir(os.getcwd()+"/train_images_sorted/" + str(x))
        i = 0
        total = []
        for filename in os.listdir(os.getcwd()):
            i += 1
            image = Extractor.getImage(filename)
            matrix = Extractor.ImageToMatrix(image)
            data = black_percentage(matrix)
            if len(total) == len(data):
                total = add_grayscale(data, total)
            else:
                total = data
        array = average_percentage(total ,i)
        print("Digit " + str(x) + " is complete")
        img = Image.fromarray(array.astype(np.uint8))
        img.save("average.tif")
        # img.show()

def black_percentage(grayscale):
    """
    NumPy 2D array -> same dimension NumPy 2D array
    
    returns black % for each pixel, 1 is completely black, 0 is white
    """
    
    r = np.zeros((grayscale.shape[0], grayscale.shape[1]), dtype=int)
    for row in range(len(grayscale)):
        for col in range(len(grayscale[row])):
            r[row][col] = grayscale[row][col]
    return r

def average_percentage(sum_grayscale, num):
    r = np.zeros((sum_grayscale.shape[0], sum_grayscale.shape[1]), dtype=int)
    for row in range(len(sum_grayscale)):
        for col in range(len(sum_grayscale[row])):
            r[row][col] = (sum_grayscale[row][col])//num
    return r    

def add_grayscale(g1, g2):
    r = np.zeros((g1.shape[0], g1.shape[1]), dtype=int)
    for row in range(len(g1)):
        for col in range(len(g1[row])):
            r[row][col] = g1[row][col] + g2[row][col]
    return r      
    

if __name__ == "__main__":
    loop_images();

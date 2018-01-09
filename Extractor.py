import tkinter as tk
from tkinter import filedialog
from PIL import Image
import numpy as numpy

def getImage():
    """Returns a two dimensional array of a chosen image's pixels"""
    
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    file = open(file_path)
    
    print(file.name)
    
    try:
        image = Image.open(file.name, 'r')
    except Exception as e:
        print(e)
    
    return image

def ImageToArray(image):
    width, height = image.size
    pixel_values = list(image.getdata())
    color_array = []
    for h in range(height):
        color_array.append([])
        for w in range(width):
            color_array[h].append(pixel_values[h*width+w])
    return color_array


def ImageToMatrix(image):
    return numpy.asarray(image)


if __name__ == "__main__":
    image = getImage()
    #matrix = ImageToMatrix(image)
    #print(matrix)
    array = ImageToArray(image)
    for row in array:
        print(row)
    exit = input("Type any enter to exit")

import tkinter as tk
from tkinter import filedialog
from PIL import Image

def getImageRGB():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    file = open(file_path)
    
    print(file.name)
    
    try:
        im = Image.open(file.name, 'r')
        width, height = im.size
        pixel_values = list(im.getdata())

        for value in pixel_values:
            print(value)
    except:
        print("Choose an actual image file")
        try:
            getImageRGB()
        except:
            pass
    


if __name__ == "__main__":
    getImageRGB()
    exit = input("Type any enter to exit")

from tkinter import *
from PIL import Image
from PIL import ImageDraw
import numpy as np
import Extractor
sys.path.append('-Averaged Approach-')
import Tester
sys.path.append('..')



array = np.full((280,280),255)
img = Image.fromarray(array.astype(np.uint8))
img.save("output.tif")
class Paint():

    def __init__(self):
        self.root = Tk()
        self.root.title("Character Recogniser")
        
        self.canvas = Canvas(self.root, bg='white', width=280, height=280)
        self.canvas.grid(row=0, columnspan=2)
           
        self.clear_button = Button(self.root, text='Clear', command=self.clear)
        self.clear_button.grid(row=1, column=1)
        
        self.predict_button = Button(self.root, text='Predict!', command=self.predict)
        self.predict_button.grid(row=2, column=1)
        
        self.result_label = Label(self.root, text="Draw something!")
        self.result_label.grid(row=1, column=0)        
        
        self.old_x, self.old_y = None, None
        self.canvas.bind('<B1-Motion>', self.paint)
        self.canvas.bind('<ButtonRelease-1>', self.release)        
        self.root.mainloop()        
        
    def clear(self):
        self.canvas.delete("all")
        self.result_label['text'] = 'Draw something!'
        global img
        array = np.full((280,280),255)
        img = Image.fromarray(array.astype(np.uint8))
        # img.save("output.tif")
        
    def predict(self):
        #global array
        #img = Image.fromarray(array.astype(np.uint8))
        #img.save("output.tif")
        global img
        img = img.resize((28, 28),Image.BILINEAR)
        img.save("output.tif")
        darken(img)
        
        prediction = Tester.test_one(img)
        self.result_label['text'] = 'Prediction: ' + str(prediction)        
                                  
    def paint(self, event):
        if self.old_x and self.old_y:
            self.canvas.create_line(self.old_x, self.old_y, event.x, event.y, width=10, fill='black', capstyle=ROUND)
            
            global img
            draw = ImageDraw.Draw(img)
            draw.line((self.old_x,self.old_y, event.x,event.y), fill=0, width = 1)
            
        self.old_x, self.old_y = event.x, event.y
        
    
    def release(self, event):
        self.old_x, self.old_y = None, None
    
def darken(img):
    
    matrix = Extractor.ImageToMatrix(img)
    matrix.setflags(write=1)
    darkest = 255
    for row in range(len(matrix)):
        for col in range(len(matrix[row])):
            if matrix[row][col] < darkest:
                darkest = matrix[row][col]
    # enhace the image     
    for row in range(len(matrix)):
        for col in range(len(matrix[row])):
            if matrix[row][col]!=255:
                matrix[row][col] -= darkest
    img = Image.fromarray(matrix.astype(np.uint8))
    img.save("enhaced.tif")        
    
            
if __name__ == '__main__':
    Paint()

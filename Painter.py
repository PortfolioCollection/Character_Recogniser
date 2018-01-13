from tkinter import *

class Paint():

    def __init__(self):
        self.root = Tk()
        self.c = Canvas(self.root, bg='white', width=280, height=280)
        self.c.grid(row=5, columnspan=5)
        self.old_x, self.old_y = None, None
        self.c.bind('<B1-Motion>', self.paint)
        self.c.bind('<ButtonRelease-1>', self.reset)        
        self.root.mainloop()        
        
    def paint(self, event):
        if self.old_x and self.old_y:
            self.c.create_line(self.old_x, self.old_y, event.x, event.y, width=10, fill='black', capstyle=ROUND, smooth=TRUE)
        self.old_x, self.old_y = event.x, event.y        
    
    def reset(self, event):
        self.old_x, self.old_y = None, None


if __name__ == '__main__':
    Paint()
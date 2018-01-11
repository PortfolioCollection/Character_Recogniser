import Extractor
import os


def loop_images():
    os.chdir(os.getcwd()+"/test_images")
    for filename in os.listdir(os.getcwd()):
        image = Extractor.getImage(filename)
        matrix = Extractor.ImageToMatrix(image)
        print(matrix)
        


if __name__ == "__main__":
    loop_images();

import os

def read():
    os.chdir('..')
    f = open("mnist-train-labels.txt", "r").read().split("\n")
    os.makedirs(str(os.getcwd()) + "/train_images_sorted")
    os.makedirs(str(os.getcwd()) + "/train_images_sorted/0")
    os.makedirs(str(os.getcwd()) + "/train_images_sorted/1")
    os.makedirs(str(os.getcwd()) + "/train_images_sorted/2")
    os.makedirs(str(os.getcwd()) + "/train_images_sorted/3")
    os.makedirs(str(os.getcwd()) + "/train_images_sorted/4")
    os.makedirs(str(os.getcwd()) + "/train_images_sorted/5")
    os.makedirs(str(os.getcwd()) + "/train_images_sorted/6")
    os.makedirs(str(os.getcwd()) + "/train_images_sorted/7")
    os.makedirs(str(os.getcwd()) + "/train_images_sorted/8")
    os.makedirs(str(os.getcwd()) + "/train_images_sorted/9")
    i = 0
    percent = 1
    for line in f:
        i += 1
        os.rename(str(os.getcwd()) + "/train_images/" + file_name(i), str(os.getcwd()) + "/train_images_sorted/" + str(line) + "/" + file_name(i))
        if i % 600 == 0:
            print(str(percent) + "%")
            percent += 1

def file_name(num):
    r = str(num)
    while len(r) < 5:
        r = "0" + r
    return r + ".tif"

if __name__ == "__main__":
    read();

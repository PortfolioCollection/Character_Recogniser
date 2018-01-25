import os
from PIL import ImageOps
from PIL import Image
import sys
import numpy as np
sys.path.append('../_Core Functions_')
import Extractor

def test(answer_array,index,filename):
    NN = 7

    test_image = Extractor.getImage(filename)
    test_image = crop_image(test_image)
    test = Extractor.ImageToMatrix(test_image)
    os.chdir('..')
    os.chdir(os.getcwd()+"/"+"train_images_sorted"+"/")


    scores = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[]}
    for x in range(10):
        os.chdir(os.getcwd()+"/" + str(x) + "/")
        
        for filename in os.listdir(os.getcwd()):
            trained_image = Extractor.getImage(filename)
            #trained_image = crop_image(trained_image)
            #trained_image = ImageOps.fit(trained_image, (len(test[0]),len(test)), Image.ANTIALIAS)
            trained = Extractor.ImageToMatrix(trained_image)
            scores = add_score(scores, x, matching_score(test, trained), NN)
            #print("done a file")
        os.chdir('..')


    print(scores)
    os.chdir('..')
    os.chdir(os.getcwd()+"/test_images/")
    if answer_array[index] == predict(scores, NN):
        return 1
    return 0

def predict(scores, NN):
    ranked = []
    for x in range(NN):
        best = (69, 255*28*28)
        for d in scores:
            for i in range(len(scores[d])):
                if scores[d][i] < best[1]:
                    best = (d, i)
        ranked.append(best[0])
        del scores[best[0]][best[1]]
    return max(set(ranked), key=ranked.count)

    
def add_score(scores, digit, element, NN):
    total = 0
    worst = (69,0)
    for d in scores:
        for i in range(len(scores[d])):
            total += 1
            if scores[d][i] >= worst[1]:
                worst = (d, i)

    if total >= NN:
        del scores[worst[0]][worst[1]]
    scores[digit].append(element)
    
    return scores
            

def test_one(img):

    FOLDER_NAME = "-Averaged Approach-"
    test = Extractor.ImageToMatrix(img)
    os.chdir(os.getcwd()+"/"+FOLDER_NAME+"/")
    os.chdir(os.getcwd()+"/trained_digits/")
    scores = []
    
    for x in range(10):
        trained_filename = str(x) + ".tif"    
        
        trained_image = Extractor.getImage(trained_filename)
        trained = Extractor.ImageToMatrix(trained_image)            
        
        scores.append(matching_score(test, trained))
    
    guess = scores.index(min(scores))
    confidence = (1 - min(scores)/(255*28*28)) * 100
    os.chdir('..')
    os.chdir('..')   
    return guess
    
def matching_score(test, digit):
    score = 0
    for row in range(len(test)):
        for col in range(len(test[row])):
            if not test[row][col] == 255 and not digit[row][col] == 255:
                invert_test = 255 - test[row][col]
                invert_memory = 255 - digit[row][col]
                score += abs(invert_memory - invert_test)
    return score

def run_test(num_tests=10000):
    STOP_AT = min(num_tests,10000)
    PERCENTILE = STOP_AT/100
    
    answer_array = []
    os.chdir('..')
    answers = open("mnist-test-labels.txt", "r")
    
    index = 0
    for line in answers:
        answer_array.append(int(line.strip()))

    #print(answer_array)
    index = 0
    correct = 0
    percent = 0
    os.chdir(os.getcwd()+"/test_images/")
    for filename in os.listdir(os.getcwd()):
        correct += test(answer_array, index, filename)
        index+=1
        if index % PERCENTILE == 0:
            print(str(percent) + "%")
            percent += 1
        if index == STOP_AT:
            break

    print(str(correct/index*100)+"% correct")

def crop_image(image):
    matrix = Extractor.ImageToMatrix(image)
    # print(matrix)
    top,bottom,left,right = get_image_bounds(matrix)
    # print(get_image_bounds(matrix))
    img = image.crop((left, top, right + 1, bottom + 1))
       
    return img  


def get_image_bounds(matrix):
    LIMIT = 200
    # Top bound
    top = 69
    found = False
    i = 0
    while found == False:
        for x in range(len(matrix[i])):
            if matrix[i][x] < LIMIT:
                top = i
                found = True
        i += 1
    
    # Bottom bound
    bottom = 69
    found = False
    i = 27
    while found == False:
        for x in range(len(matrix[i])):
            if matrix[i][x] < LIMIT:
                bottom = i
                found = True
        i -= 1
    
    # Left bound
    left = 69
    found = False
    i = 0
    while found == False:
        for x in range(len(matrix[i])):
            if matrix[x][i] < LIMIT:
                left = i
                found = True
        i += 1
       
    # Right bound
    right = 69
    found = False
    i = 27
    while found == False:
        for x in range(len(matrix[i])):
            if matrix[x][i] < LIMIT:
                right = i
                found = True
        i -= 1

    return (top, bottom, left, right)
    

if __name__ == "__main__":
    run_test(2)

    

    

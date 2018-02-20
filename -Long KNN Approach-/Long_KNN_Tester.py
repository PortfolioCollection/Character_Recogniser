import os
from PIL import ImageOps
from PIL import Image
import sys
import numpy as np
sys.path.append('../_Core Functions_')
import Extractor
import Manipulate_Image
import time

def test(answer_array,index,filename):
    NN = 3
    count = 0

    STOP = 100
    
    test_image = Extractor.getImage(filename)
    test_image = Manipulate_Image.crop_image(test_image)
    test = Extractor.ImageToMatrix(test_image)
    os.chdir('..')
    os.chdir(os.getcwd()+"/"+"train_images_sorted"+"/")


    scores = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[]}
    for x in range(10):
        os.chdir(os.getcwd()+"/" + str(x) + "/")
        for filename in os.listdir(os.getcwd()):
            if count == STOP: break
            trained_image = Extractor.getImage(filename)
            trained_image = Manipulate_Image.crop_image(trained_image)
            trained_image = ImageOps.fit(trained_image, (len(test[0]),len(test)), Image.ANTIALIAS)
            trained = Extractor.ImageToMatrix(trained_image)
            scores = add_score(scores, x, matching_score(test, trained), NN)
            #print("done a file")
            count+=1
        count = 0
        os.chdir('..')


    # print(scores)
    os.chdir('..')
    os.chdir(os.getcwd()+"/test_images/")


    guess = predict(scores)
    #print("Actual: "+str(answer_array[index])+" Best: "+str(guess))
    if answer_array[index] == guess:
        return 1
    return 0

def predict(scores):
    # print(scores)
    ranked = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[]}
    for d in scores:
        scores[d].sort()
        ranked[d].append(len(scores[d]))
        if len(scores[d]) > 0:
            ranked[d].append(scores[d][0])
    best = None
    for d in ranked:
        if best is None and len(ranked[d]) > 1:
            best = [d, ranked[d][1]]
        if best is not None and len(ranked[d]) > 1:
            if ranked[d][0] > ranked[best[0]][0]:
                best = [d, ranked[d][1]]
            if ranked[d][0] == ranked[best[0]][0]:
                #print("Ranked: "+str(ranked)+" d: "+str(d)+" best[0]: "+str(best[0]))
                if ranked[d][1] < ranked[best[0]][1]:
                    best = [d, ranked[d][1]]
    return best[0]


    
def add_score(scores, digit, element, NN):
    # print(scores)
    # print((digit, element))
    total = 0
    worst = None
    for d in scores:
        for i in range(len(scores[d])):
            total += 1
            if worst is None or scores[d][i] >= scores[worst[0]][worst[1]]:
                worst = [d, i]
    # print(worst)
    if total < NN:
        scores[digit].append(element)
    if total >= NN and element <= scores[worst[0]][worst[1]]:
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
            if not test[row][col] == 255 or not digit[row][col] == 255:
                invert_test = 255 - test[row][col]
                invert_memory = 255 - digit[row][col]
                score += abs(invert_memory - invert_test)
    return score

def run_test(num_tests=10000):
    file = open('status.txt', 'w')
    file.write(str(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))+"\n")
    file.flush()
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
    percent = 1
    os.chdir(os.getcwd()+"/test_images/")
    for filename in os.listdir(os.getcwd()):
        correct += test(answer_array, index, filename)
        index+=1
        if index % PERCENTILE == 0:
            print(str(percent) + "%")
            percent += 1
        file.write(str(index)+": "+str(round(correct/index*100,2))+"%\n")
        file.flush()
        if index == STOP_AT:
            break
    file.write("done")
    file.flush()
    file.close()
    print(str(correct/index*100)+"% correct")

if __name__ == "__main__":
    start = time.time()
    run_test(100)
    print("It took " + str(time.time()-start) + "seconds")
    

    

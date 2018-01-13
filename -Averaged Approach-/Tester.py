import os
import sys
sys.path.append('..')
import numpy as np
import Extractor


def test(answer_array,index,filename):

    FOLDER_NAME = "-Averaged Approach-"

    
    test_image = Extractor.getImage(filename)
    test = Extractor.ImageToMatrix(test_image)
    os.chdir('..')
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
    os.chdir(os.getcwd()+"/test_images/")
    #print(scores)
    #print("==================================================================")
    #print("Perdict digit: " + str(guess) + ", with confidence " + str(confidence) + "%")
    if answer_array[index] == guess:
        return 1
    return 0


    
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

if __name__ == "__main__":
    run_test(500)

    

    

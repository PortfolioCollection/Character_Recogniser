import os
import sys
import numpy as np
sys.path.append('../_Core Functions_')
import Extractor
import KNN_Trainer
import re

def test(answer_array,index,filename):
    FOLDER_NAME = "-KNN Approach-"
    
    
    lr = KNN_Trainer.read_image(filename)
    lean = KNN_Trainer.record_left_right(lr)
    segments = KNN_Trainer.record_segment(lr)


    os.chdir('..')
    os.chdir(os.getcwd()+"/"+FOLDER_NAME+"/")
    
    neighbors = open("save.txt")

    best_score = 100
    optimal_number = -1
    
    for line in neighbors:
        match = "line ([0-9]*): lean\(([0-9].[0-9]*)\) segment\(([0-9].[0-9]*)\) class\(([0-9])\)"
        string = re.match(match, line)
        train_line,train_lean,train_segments,train_number = string.group(1),string.group(2),string.group(3),string.group(4)
        #print(train_line)
        score = abs(lean-float(train_lean))+abs(segments-float(train_segments))
        if score < best_score:
            best_score = score
            optimal_number = train_number
    print("Score: "+str(score))
    print("Optimal Number: "+str(optimal_number))
                
        

    os.chdir('..')
    os.chdir(os.getcwd()+"/test_images/")
    print("Answer: "+str(answer_array[index]))
    if answer_array[index] == optimal_number:
        return 1
    return 0
    
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
        break #comment this out later

    print(str(correct/index*100)+"% correct")

if __name__ == "__main__":
    run_test(1)

    

    

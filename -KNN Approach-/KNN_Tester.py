#--------Hopping-------#
import os
import sys
sys.path.append('../_Core Functions_')
import Hop
#----CUSTOM CLASSES-----#
import Extractor
import KNN_Trainer
#---SUPPORT LIBRARIES---#
import numpy as np
import time
import re
import time

def test_image(answer_array,index,filename):
    image = Extractor.getImage(filename)
    optimal_number = test_one(image)
    Hop.go_to_TestImages()
    if answer_array[index] == int(optimal_number):
        return 1
    return 0

def test_one(image):
    FOLDER_NAME = "-KNN Approach-"
    test = Extractor.ImageToMatrix(image)

    Hop.go_to_approach("/"+FOLDER_NAME)
    
    best_score = 100
    optimal_number = -1
    
    grayscale = Extractor.ImageToMatrix(image)
    r = np.zeros((grayscale.shape[0], grayscale.shape[1]), dtype=int)
    
    lr = KNN_Trainer.read_image(grayscale)
    lean = KNN_Trainer.record_left_right(lr)
    segments = KNN_Trainer.record_segment(lr)
    outside = KNN_Trainer.inside_outside(lr,grayscale)

    neighbors = open("save.txt")
    
    for line in neighbors:
        match = "line ([0-9]*): lean\(([0-9].[0-9]*)\) segment\(([0-9].[0-9]*)\) outside\(([0-9].[0-9]*)\) class\(([0-9])\)"
        string = re.match(match, line)
        train_line,train_lean,train_segments,train_outside,train_number = string.group(1),string.group(2),string.group(3),string.group(4),string.group(5)
        score = abs(lean-float(train_lean))+abs(segments-float(train_segments))
        if score < best_score:
            best_score = score
            optimal_number = train_number
    return optimal_number
    
    
def test_loop(num_tests=10000):
    file = open('status.txt', 'w')
    file.write(str(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))+"\n")
    file.flush()
    STOP_AT = min(num_tests,10000)
    PERCENTILE = STOP_AT/100
    
    answer_array = []
    Hop.go_to_home()
    answers = open("mnist-test-labels.txt", "r")
    
    index = 0
    for line in answers:
        answer_array.append(int(line.strip()))

    index = 0
    correct = 0
    percent = 1
    Hop.go_to_TestImages()
    start_time = time.time()
    for filename in os.listdir(os.getcwd()):
        correct += test_image(answer_array, index, filename)
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
    duration = (time.time()-start_time)
    print("Seconds:"+str(duration))
    print(str(correct/index*100)+"% correct")

    

if __name__ == "__main__":
    os.chdir("..")
    Hop.set_project_path()
    Hop.go_to_approach("/-KNN Approach-")
    test_loop(50)

    

    

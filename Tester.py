import os
import numpy as np
import Extractor


def test():    
    test_image = Extractor.getImage("test.tif")
    test = Extractor.ImageToMatrix(test_image)    

    os.chdir(os.getcwd()+"/trained_digits/")
    
    scores = []
    for x in range(10):
        trained_filename = str(x) + ".tif"    
        
        trained_image = Extractor.getImage(trained_filename)
        trained = Extractor.ImageToMatrix(trained_image)            
        
        scores.append(matching_score(test, trained))
    
    guess = scores.index(min(scores))
    confidence = (1 - min(scores)/(255*28*28)) * 100
    
    print(scores)
    print("==================================================================")
    print("Perdict digit: " + str(guess) + ", with confidence " + str(confidence) + "%")

def matching_score(test, digit):
    score = 0
    for row in range(len(test)):
        for col in range(len(test[row])):
            score += abs(test[row][col] - digit[row][col])
    return score          
    


if __name__ == "__main__":
    test();

import sys
import os
sys.path.append(os.getcwd())
sys.path.append(os.getcwd()+"/_Core Functions_")
sys.path.append(os.getcwd()+"/-Averaged Approach-")
sys.path.append(os.getcwd()+"/-KNN Approach-")
import Averaged_Approach
import KNN_Approach

def average_approach(NUM_TESTS):
    os.chdir('-Averaged Approach-')
    #print("average")
    Averaged_Approach.run_approach(NUM_TESTS)


def knn_approach(NUM_TESTS):
    os.chdir('-KNN Approach-')
    print("knn")
    KNN_Approach.run_approach(NUM_TESTS)


if __name__ == "__main__":
    NUM_TESTS = 500
    approach = input("(1) Averaged Approach \n\
(2) KNN Approach\n")
    options = {'1' : average_approach,
               '2': knn_approach}
    try:
        options[approach](NUM_TESTS)
    except Exception as e:
        print("Enter an actual option")

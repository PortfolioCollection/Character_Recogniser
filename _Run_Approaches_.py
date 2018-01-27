#--------Hopping-------#
import sys
import os
sys.path.append(os.getcwd()+"/_Core Functions_")
import Hop
#----CUSTOM CLASSES-----#
Hop.set_project_path()
Hop.go_to_core()
from Painter import*
Hop.go_to_home()

def average_approach(NUM_TESTS):
    sys.path.append(os.getcwd()+"/-Averaged Approach-")
    import Averaged_Approach
    os.chdir('-Averaged Approach-')
    Averaged_Approach.run_approach(NUM_TESTS)
    #Painter(Averaged_Approach.test_one())

def knn_approach(NUM_TESTS):
    #sys.path.append(os.getcwd()+"/-KNN Approach-")
    #import KNN_Approach
    #os.chdir('-KNN Approach-')
    #KNN_Approach.run_approach(NUM_TESTS)
    Hop.go_to_core()
    paint = Paint()


if __name__ == "__main__":
    #print(os.getcwd())
    NUM_TESTS = 500
    approach = input("(1) Averaged Approach \n\
(2) KNN Approach\n")
    options = {'1' : average_approach,
               '2': knn_approach}
    #try:
    options[approach](NUM_TESTS)
    #except Exception as e:
    #    print("Enter an actual option")

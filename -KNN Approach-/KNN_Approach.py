#--------Hopping-------#
import os
import sys
sys.path.append('../_Core Functions_')
import Hop
#----CUSTOM CLASSES-----#
import KNN_Tester
import KNN_Trainer

def run_approach(num):
    KNN_Tester.test_loop(num)

def train_approach():
    KNN_Trainer.train_images()

#if __name__ == "__main__":
    #os.chdir("..")
    #Hop.set_project_path()
    #run_approach(50)

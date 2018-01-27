#--------Hopping-------#
import os
import sys
sys.path.append('../_Core Functions_')
import Hop
#----CUSTOM CLASSES-----#
import Averaged_Tester
import Averaged_Trainer

def run_approach(num):
    Averaged_Tester.run_test(num)

def train_approach():
    Averaged_Trainer.train_images()

if __name__ == "__main__":
    os.chdir("..")
    Hop.set_project_path()
    Averaged_Tester.run_test(500)

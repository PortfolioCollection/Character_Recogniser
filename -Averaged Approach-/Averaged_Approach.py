import sys
import os
import Averaged_Tester
import Averaged_Trainer

def run_approach(num):
    Averaged_Tester.run_test(num)

def train_approach():
    Averaged_Trainer.train_images()

if __name__ == "__main__":
    Averaged_Tester.run_test(500)

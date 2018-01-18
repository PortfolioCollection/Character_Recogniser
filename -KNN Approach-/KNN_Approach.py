import sys
import os
import KNN_Tester
import KNN_Trainer

def run_approach(num):
    KNN_Tester.run_test(num)

def train_approach():
    KNN_Trainer.train_images()

if __name__ == "__main__":
    run_approach(500)

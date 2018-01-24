import sys
import os
import Long_KNN_Tester
import Long_KNN_Trainer

def run_approach(num):
    Long_KNN_Tester.run_test(num)

def train_approach():
    Long_KNN_Trainer.train_images()

if __name__ == "__main__":
    Long_KNN_Tester.run_test(500)

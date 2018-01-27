import os

PROJECT_PATH = ""
CORE_FUNCTIONS = "/_Core Functions_"
TEST_IMAGES = "/test_images"
TRAIN_IMAGES = "/train_images_sorted"

def set_project_path():
    global PROJECT_PATH
    PROJECT_PATH = os.getcwd()

def go_to_home():
    global PROJECT_PATH
    os.chdir(PROJECT_PATH)

def go_to_core():
    global PROJECT_PATH
    global CORE_FUNCTIONS
    os.chdir(PROJECT_PATH+CORE_FUNCTIONS)

def go_to_approach(APPROACH):
    global PROJECT_PATH
    os.chdir(PROJECT_PATH+APPROACH)

def go_to_TestImages():
    global PROJECT_PATH
    global TEST_IMAGES
    os.chdir(PROJECT_PATH+TEST_IMAGES)

def go_to_TrainImages():
    global PROJECT_PATH
    global TRAIN_IMAGES
    os.chdir(PROJECT_PATH+TRAIN_IMAGES)


def cwd():
    return os.getcwd()

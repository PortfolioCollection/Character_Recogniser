#--------Hopping-------#
import Hop
import os
import sys

#----CUSTOM CLASSES-----#
os.chdir("..")
Hop.set_project_path()
sys.path.append(os.getcwd()+"/-Averaged Approach-")
sys.path.append(os.getcwd()+"/-KNN Approach-")
sys.path.append(os.getcwd()+"/-Long KNN Approach-")
Hop.go_to_core()

#----SUPPORT LIBRARIES-----#
import re

def read_status(approach, number, accuracy, full):
    Hop.go_to_approach(approach)
    try:
        file = open("status.txt","r")
        lines = file.readlines()
        last_line = "no lines to show"
        for line in lines[::-1]:
            if line == "done":
                continue
            else:
                last_line = line
                break

        
        prog = re.compile("([0-9]*): ([0-9]*.[0-9]*)%")
        result = prog.match(last_line)

        num = ""
        acc = ""
        
        
        if number:
            num=result[1]
        if accuracy:
            acc=result[2]
        if full:
            for line in lines[1:]:
                if line == "done":
                    break
        return (num, acc)
    except Exception as e:
        print(e)

if "__main__" == __name__:
    print(read_status("/-KNN Approach-",True,True,False))

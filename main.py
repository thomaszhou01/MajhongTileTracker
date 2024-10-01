import os
import cv2
import collections
import json
from tkinter import *
from tileDetector import *


def getTiles():
    compare = []
    cwd = os.getcwd()
    dir_list = os.listdir(cwd + "/tiles")

    for i in dir_list:
        compareImg = cv2.imread("tiles/" + i, cv2.COLOR_BGR2GRAY)
        compare.append((i, compareImg))
    return compare


def main():
    compare = getTiles()
    tracker = {}
    with open("data.json") as f:
        tracker = json.load(f)

    print(tracker)

    for i in runDetection(compare=compare, region=(0, 800, 1980, 1080)):
        tracker[i] = tracker.get(i, 0) + 1
        print(i)

    with open("data.json", "w") as outfile:
        json.dump(tracker, outfile)


if __name__ == "__main__":
    main()

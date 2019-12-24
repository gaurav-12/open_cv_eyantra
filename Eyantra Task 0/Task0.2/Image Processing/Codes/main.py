import cv2
# import numpy as np
import csv
import os


def partA():
    with open('../Generated/stats.csv', 'w') as statsFile:
        for entry in (list(os.scandir('../Images/'))):
            toWrite = [entry.name]

            image = cv2.imread(entry.path)

            width, height, channel = image.shape
            toWrite.append(height)
            toWrite.append(width)
            toWrite.append(channel)

            toWrite.extend(image[round((width / 2)) - 1, round((height / 2)) - 1])

            writer = csv.writer(statsFile)
            writer.writerow(toWrite)


def partB():
    for entry in (list(os.scandir('../Images/'))):
        if "cat" in entry.name:
            image = cv2.imread(entry.path)
            image[:, :, 0] = 0
            image[:, :, 1] = 0
            cv2.imwrite('../Generated/cat_red.jpg', image)


def partC():
    for entry in (list(os.scandir('../Images/'))):
        if "flowers" in entry.name:
            image = cv2.imread(entry.path)

            alphaImage = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
            alphaImage[:, :, 3] = 255/2

            cv2.imwrite('../Generated/flowers_alpha.png', alphaImage)


def partD():
    for entry in (list(os.scandir('../Images/'))):
        if "horse" in entry.name:
            image = cv2.imread(entry.path)
            b, g, r = cv2.split(image)

            hsvImage = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hsvImage[:, :, 2] = ((0.3 * r) + (0.59 * g) + (0.11 * b))
            rgbImage = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

            cv2.imwrite('../Generated/horse_gray.jpg', rgbImage)


partA()
partB()
partC()
partD()

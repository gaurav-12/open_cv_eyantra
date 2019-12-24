import cv2
import numpy as np
import os


def partA():
    vid = cv2.VideoCapture('../Videos/RoseBloom.mp4')

    vid.set(cv2.CAP_PROP_POS_FRAMES, 125)

    ret, image = vid.read()

    if ret:
        cv2.imwrite('../Generated/frame_as_6.jpg', image)

    vid.release()
    cv2.destroyAllWindows()


def partB():
    vid = cv2.VideoCapture('../Videos/RoseBloom.mp4')

    vid.set(cv2.CAP_PROP_POS_FRAMES, 125)

    ret, image = vid.read()

    if ret:
        image[:, :, 0] = 0
        image[:, :, 1] = 0
        cv2.imwrite('../Generated/frame_as_6_red.jpg', image)

    vid.release()
    cv2.destroyAllWindows()


partA()
partB()

###############################################################################
## Author: Team Supply Bot
## Edition: eYRC 2019-20
## Instructions: Do Not modify the basic skeletal structure of given APIs!!!
###############################################################################


######################
## Essential libraries
######################
import cv2
import numpy as np
import os
import math
import csv
import cv2.aruco as aruco
from aruco_lib import *
import copy

########################################################################
## using os to generalise Input-Output
########################################################################
codes_folder_path = os.path.abspath('.')
images_folder_path = os.path.abspath(os.path.join('..', 'Videos'))
generated_folder_path = os.path.abspath(os.path.join('..', 'Generated'))


############################################
## Build your algorithm in this function
## ip_image: is the array of the input image
## imshow helps you view that you have loaded
## the corresponding image
############################################

def detectAruco(frame):
    robot_state = {}
    det_aruco_list = detect_Aruco(frame)
    if det_aruco_list:
        frame = mark_Aruco(frame, det_aruco_list)
        robot_state = calculate_Robot_State(frame, det_aruco_list)

    cv2.imshow("Frame", frame)

    return frame, list(robot_state.values())[0]


def blurEdge(img, d=31):
    hei, wei = img.shape[:2]
    imgPad = cv2.copyMakeBorder(img, d, d, d, d, cv2.BORDER_WRAP)
    imgBlur = cv2.GaussianBlur(imgPad, (2 * d + 1, 2 * d + 1), -1)[d:-d, d:-d]

    y, x = np.indices((hei, wei))
    dist = np.dstack([x, wei - x - 1, y, hei - y - 1]).min(-1)
    w = np.minimum(np.float32(dist) / d, 1.0)

    return (img * w) + imgBlur * (1 - w)


def motKernel(angle, d, sz=65):
    kern = np.ones((1, d), np.float32)
    c, s = np.cos(angle), np.sin(angle)
    A = np.float32([[c, -s, 0], [s, c, 0]])
    sz2 = sz // 2
    A[:, 2] = (sz2, sz2) - np.dot(A[:, :2], ((d - 1) * 0.5, 0))
    kern = cv2.warpAffine(kern, A, (sz, sz), flags=cv2.INTER_CUBIC)

    return kern


def process(ip_image):
    img = ip_image

    filtChannels = []

    for ch in cv2.split(img):
        img = np.float32(ch) / 255.0

        img = blurEdge(img)
        imgFT = cv2.dft(img, flags=cv2.DFT_COMPLEX_OUTPUT)

        angle = np.deg2rad(90)
        length = 20
        noise = 10 ** (-0.1 * 25)

        # PSF through Motion Kernel, of 20x20 at 90deg angle
        psf = motKernel(angle, length)
        cv2.imshow('psf', psf)

        # Normalize PSF
        psf /= psf.sum()

        psfPad = np.zeros_like(img)
        kh, kw = psf.shape
        psfPad[:kh, :kw] = psf
        psfFt = cv2.dft(psfPad, flags=cv2.DFT_COMPLEX_OUTPUT, nonzeroRows=kh)
        psfSq = (psfFt ** 2).sum(-1)
        finPSF = psfFt / (psfSq + noise)[..., np.newaxis]

        # Multiply Fourier Transform of PSF and Image
        res = cv2.mulSpectrums(imgFT, finPSF, 0)
        res = cv2.idft(res, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
        res = np.roll(res, -kh // 2, 0)
        res = np.roll(res, -kw // 2, 1)

        # Append this channel's result to the list
        filtChannels.append(res)

    # Merge all the channels to get the colored image
    finalImg = cv2.merge(filtChannels, ip_image)

    # Set Brightness and Contrast
    finalImg = cv2.addWeighted(finalImg, 2, finalImg, 0, 0)

    cv2.imshow("Final Image", finalImg)

    # Next, convert image to uint8 and find the aruco
    finalImg = cv2.convertScaleAbs(finalImg, alpha=255.0)
    finalImg, arucoInfo = detectAruco(finalImg)

    # Writing file with id and angle on it.
    cv2.imwrite("../Generated/aruco_with_id.png", finalImg)

    cv2.waitKey(0)

    return finalImg, arucoInfo


####################################################################
## The main program which provides read in input of one image at a
## time to process function in which you will code your generalized
## output computing code
## Do not modify this code!!!
####################################################################
def main(val):
    ################################################################
    ## variable declarations
    ################################################################
    i = 1
    ## reading in video 
    cap = cv2.VideoCapture(images_folder_path + "/" + "ArUco_bot.mp4")
    ## getting the frames per second value of input video
    fps = cap.get(cv2.CAP_PROP_FPS)
    ## getting the frame sequence
    frame_seq = int(val) * fps
    ## setting the video counter to frame sequence
    cap.set(1, frame_seq)
    ## reading in the frame
    ret, frame = cap.read()
    ## verifying frame has content
    print(frame.shape)
    ## display to see if the frame is correct
    cv2.imshow("window", frame)
    cv2.waitKey(0);
    ## calling the algorithm function
    op_image, aruco_info = process(frame)
    ## saving the output in  a list variable
    line = [str(i), "Aruco_bot.jpg", str(aruco_info[0]), str(aruco_info[3])]
    ## incrementing counter variable
    i += 1
    ## verifying all data
    print(line)
    ## writing to angles.csv in Generated folder without spaces
    with open(generated_folder_path + "/" + 'output.csv', 'w') as writeFile:
        print("About to write csv")
        writer = csv.writer(writeFile)
        writer.writerow(line)
    ## closing csv file    
    writeFile.close()


############################################################################################
## main function
############################################################################################
if __name__ == '__main__':
    main(input("time value in seconds:"))

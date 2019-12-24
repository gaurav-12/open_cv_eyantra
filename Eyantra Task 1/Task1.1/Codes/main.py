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

########################################################################
## using os to generalise Input-Output
########################################################################
codes_folder_path = os.path.abspath('.')
images_folder_path = os.path.abspath(os.path.join('..', 'Images'))
generated_folder_path = os.path.abspath(os.path.join('..', 'Generated'))


############################################
## Build your algorithm in this function
## ip_image: is the array of the input image
## imshow helps you view that you have loaded
## the corresponding image
############################################

def findDist(c1, c2):
    return math.sqrt((c2[0] - c1[0]) ** 2 + (c2[1] - c1[1]) ** 2)


def process(ip_image):
    angle = 0.00
    ip_image[:, :, 0] = 255
    ip_image[:, :, 1] = 255

    # Convert to Grayscale
    grIm = cv2.cvtColor(ip_image, cv2.COLOR_BGR2GRAY)

    # Find binary image of green and red dot
    # Green at 185, Red at 230
    ret, binImgR = cv2.threshold(grIm, 230, 255, cv2.THRESH_BINARY)
    ret, binImgG = cv2.threshold(grIm, 185, 255, cv2.THRESH_BINARY)

    # Apply Gaussian Blur
    gusBlurR = cv2.GaussianBlur(binImgR, (5, 5), 0)
    gusBlurG = cv2.GaussianBlur(binImgG, (5, 5), 0)

    # Find Contours of Red and Green
    (contR, hieR) = cv2.findContours(gusBlurR, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Red contour contains Centre circle and the Red Dot
    contRed, contCentre = contR[2], contR[3]  # Smallest(Red Dot), and Bigger than the Smallest(Centre Circle)
    if len(contRed) > len(contCentre):
        contCentre, contRed = contRed, contCentre

    (contG, hieG) = cv2.findContours(gusBlurG, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # print("Sizes : R- ", len(contR), " G- ", len(contG))

    # Just to draw the contours.
    cv2.drawContours(ip_image, contR, -1, (0, 0, 0), 2)
    cv2.drawContours(ip_image, contG, -1, (0, 0, 0), 2)

    # Find the Top-Left coordinates of each of circle's bounding rectangle.
    (xC, yC, widCentre, heiCentre) = cv2.boundingRect(contCentre)  # Red's third Contour is the circle in centre..

    (xR, yR, widR, heiR) = cv2.boundingRect(contRed)  # ..it's last(fourth) contour is the Red dot.
    (xG, yG, widG, heiG) = cv2.boundingRect(contG[1])  # Green's second contour is Green dot.

    # Draw lines from Centre to Red Dot, Green Dot.
    cv2.line(ip_image, (xC + widCentre // 2, yC + heiCentre // 2), (xR + widR // 2, yR + heiR // 2), (0, 0, 0), 2)
    cv2.line(ip_image, (xC + widCentre // 2, yC + heiCentre // 2), (xG + widG // 2, yG + heiG // 2), (0, 0, 0), 2)

    # Find the distance between each point, find the angle at Centre Circle's mid points, and convert the obtained
    # angle to degree.
    # Law of Cosines is used.
    coords = [(xC + widCentre / 2, yC + heiCentre / 2), (xR + widR / 2, yR + heiR / 2), (xG + widG / 2, yG + heiG / 2)]
    disCR = findDist(coords[0], coords[1])
    disCG = findDist(coords[0], coords[2])
    disRG = findDist(coords[1], coords[2])
    angle = np.rad2deg(np.arccos([((disCG ** 2) + (disCR ** 2) - (disRG ** 2)) / (2 * disCG * disCR)])[0])

    # Show the final image, with lines, and Contours
    cv2.imshow("window", ip_image)
    cv2.waitKey(0)

    # Return angle upto 2 decimal place
    return round(angle, 2)


####################################################################
## The main program which provides read in input of one image at a
## time to process function in which you will code your generalized
## output computing code
## Do not modify this code!!!
####################################################################
def main():
    ################################################################
    ## variable declarations
    ################################################################
    i = 1
    line = []
    ## Reading 1 image at a time from the Images folder
    for image_name in os.listdir(images_folder_path):
        ## verifying name of image
        print(image_name)
        ## reading in image
        ip_image = cv2.imread(images_folder_path + "/" + image_name)
        ## verifying image has content
        print(ip_image.shape)
        ## passing read in image to process function
        A = process(ip_image)
        ## saving the output in  a list variable
        line.append([str(i), image_name, str(A)])
        ## incrementing counter variable
        i += 1
    ## verifying all data
    print(line)
    ## writing to angles.csv in Generated folder without spaces
    with open(generated_folder_path + "/" + 'angles.csv', 'w', newline='') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerows(line)
    ## closing csv file
    writeFile.close()


############################################################################################
## main function
############################################################################################
if __name__ == '__main__':
    main()

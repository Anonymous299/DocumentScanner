##This program scans a document and rotates it for proper orientation

import argparse
import cv2
import imutils

##Get image as an argument
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help="path to image")
args = vars(ap.parse_args())

##Grab image from arguments and resize
image = cv2.imread(args["image"])
orig = image.copy()
image = imutils.resize(image, height = 500)

##Convert image to grayscale and blur
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

##Get edges in image
edged = cv2.Canny(blurred, 75, 200)
cv2.imshow("Edged", edged)
cv2.imshow("Orig", orig)
cv2.waitKey(0)

##Grab largest contours from edge image
cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key = cv2.contourArea, reverse=True)[:5]

##Loop through contours and find rectangular contours
for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

##    if approximated contour has 4 points, probably a rectangle
    if len(approx) == 4:
        print("Found some kind of rectangle")
        cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)
        cv2.imshow("Contours", image)
        cv2.waitKey(0)

cv2.destroyAllWindows()

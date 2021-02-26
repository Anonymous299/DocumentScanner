##This program scans a document and rotates it for proper orientation

from pyimagesearch.transform import four_point_transform
from skimage.filters import threshold_local
import argparse
import cv2
import imutils
import sys

##Get image as an argument
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help="path to image")
args = vars(ap.parse_args())

##Grab image from arguments and resize
image = cv2.imread(args["image"])
orig = image.copy()
ratio = image.shape[0] / 500.0
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
        screenCnt = approx
        break    

##If document exists, draw and show it
try:
    cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
except:
    print("No document found")
    sys.exit()
    
cv2.imshow("Contours", image)
cv2.waitKey(0)

##Rotate and translate detected document to fill the screen
warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

##Convert warped to grayscale, then threshold to obtain black and white effect
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
T = threshold_local(warped, 11, offset = 10, method = "gaussian")
warped = (warped > T).astype('uint8') * 255

##Show final scan
cv2.imshow("Document", warped)
cv2.waitKey(0)

cv2.destroyAllWindows()

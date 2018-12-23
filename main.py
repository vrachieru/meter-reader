import os
import cv2
import imutils

from imutils.perspective import four_point_transform

def extract_display(image_path):
    image = cv2.imread(image_path)
    image = imutils.resize(image, height=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 200, 255)

    # find contours in the edge map
    # then sort them by their size in descending order
    contours = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if imutils.is_cv2() else contours[1]
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for contour in contours:
        # approximate shape of the polygon
        perimeter = cv2.arcLength(contour, True)
        epsilon = 0.02 * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # if the contour has four vertices, then we have found the display
        if len(approx) == 4:
            return four_point_transform(image, approx.reshape(4, 2))


if __name__ == '__main__':
    here = os.path.dirname(__file__)
    display = extract_display(os.path.join(here, 'meter.jpg'))
    cv2.imwrite(os.path.join(here, 'display.bmp'), display)

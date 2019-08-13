import cv2
import imutils
import os

from imutils.perspective import four_point_transform

def extract_display(image):
    '''
    Detect the analog display on a meter 
    by searching for the largest rectangle shaped contour.

    :param image image: image on which to detect display
    :return image: cropped detected display
    '''
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

def extract_digits(image):
    '''
    Extract individual digits from a display by finding the 
    maximum number of contours smililarly shaped on the x axis.

    :param image image: image of analog display on which to detect digits
    :return list[image]: list of cropped detected digits
    '''
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 200, 255)

    # find contours in whole image
    contours = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    # sort bounding boxes from left to right
    boxes = [cv2.boundingRect(c) for c in contours]
    sorted_boxes = sorted(boxes, key=lambda box: box[0])

    # finding the maximum number of contours smililarly shaped on the x axis
    max_similar_aligned_boxes = []
    for box_index in range(len(sorted_boxes)):
        aligned_boxes = find_max_similar_aligned_boxes(sorted_boxes[box_index:])
        if len(aligned_boxes) > len(max_similar_aligned_boxes):
            max_similar_aligned_boxes = aligned_boxes

    # crop detected digits
    result = []
    for box in max_similar_aligned_boxes:
        (x,y,w,h) = box
        result.append(image[y:y+h, x:x+w])

    return result

def find_max_similar_aligned_boxes(boxes, treshold=40):
    '''
    Find the biggest collection of similar boxes. 

    :param list boxes: list of bounding rectangle boxes
    :param int threshold: maximum percent under which to consider two boxes as similar
    :return list: list of boxes similar to the first box in the input
    '''
    start = boxes[0]
    result = [start]
    for box in boxes[1:]:
        if box_similarity(start, box, 2) < treshold  and box_similarity(start, box, 3) < treshold:
            result.append(box)

    return result

def box_similarity(box_a, box_b, characteristic):
    '''
    Calculate the similarity for a specific characteristic between two boxes.

    :param tuple box_a: bounding rectangle box
    :param tuple box_b: bounding rectangle box
    :param int characteristic: index of the characteristic inside the tuple (x=0,y=1,w=2,h=3)
    :return float: percent of similarity between the two boxes for provided characteristic
    '''
    return abs(box_a[characteristic] - box_b[characteristic]) / max(box_a[characteristic], box_b[characteristic]) * 100


if __name__ == '__main__':
    path = lambda filename: os.path.join(os.path.dirname(__file__), filename)

    image_path = path('images/meters/gas.jpg')
    image = cv2.imread(image_path)
    
    display = extract_display(image)
    cv2.imwrite(path('display.bmp'), display)

    digits = extract_digits(display)
    for index, digit in enumerate(digits, start=1):
        cv2.imwrite(path('display_%s.bmp' % index), digit)

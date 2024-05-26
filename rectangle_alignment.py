import cv2
import numpy as np
from matplotlib import pyplot as plt
def align_rectangles(image_path):
    #reading the images
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect edges using Canny edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Find contours
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate through contours to find rectangles
    for contour in contours:
        # Find the area of each contour
        area = cv2.contourArea(contour)

        # Approximate the contour to a polygon
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        # If the contour has four vertices (a rectangle) and has a significant area, proceed
        if len(approx) == 4 and area > 1000:
            # Find the rotated rectangle enclosing the contour
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            # Calculate the angle of rotation
            angle = rect[2]

            # Rotate the image to make the rectangle straight
            rotated_image = image.copy()
            if angle < -45:
                angle += 90
            (h, w) = rotated_image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated_image = cv2.warpAffine(rotated_image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

            # Display the aligned image
            cv2.imwrite('processed_images/aligned_rectangle.png',rotated_image)

if __name__=='__main__':
    image_path = "images/unaligned_rectangle.png"
    align_rectangles(image_path)
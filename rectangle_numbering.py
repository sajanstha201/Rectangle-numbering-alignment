import cv2
import numpy as np
import matplotlib.pyplot as plt
def measure_line_length(line):
    """Measure the length of a line given its endpoints."""
    x1, y1, x2, y2 = line[0]
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# Load the image
image = cv2.imread('/Users/sajanshrestha/Downloads/rectangle.png')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply edge detection
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# Find contours
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find rectangles
rectangles = []
for contour in contours:
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
    if len(approx) == 4:
        rectangles.append(approx)

# Sort rectangles based on area
rectangles.sort(key=cv2.contourArea, reverse=True)

# Draw rectangles on the image (optional)
for rectangle in rectangles:
    cv2.drawContours(image, [rectangle], -1, (0, 255, 0), 2)

# Iterate over rectangles
for idx, rectangle in enumerate(rectangles, start=1):
    x, y, w, h = cv2.boundingRect(rectangle)
    roi = gray[y:y+h, x:x+w]  # Region of interest (ROI) inside the rectangle
    lines = cv2.HoughLinesP(roi, 1, np.pi / 180, threshold=50, minLineLength=30, maxLineGap=5)
    
    # If no lines are found, skip this rectangle
    if lines is None:
        continue
    
    # Measure the length of each line
    lengths = [(measure_line_length(line), line) for line in lines]
    
    # Sort the lines by length
    lengths.sort(key=lambda x: x[0])
    
    # Assign numbers (1 to 4) to the lines based on their order (shortest to longest)
    for i, (_, line) in enumerate(lengths[:4], start=1):
        x1, y1, x2, y2 = line[0]
        cv2.putText(image, str(i), (x + min(x1, x2), y + min(y1, y2) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

# Display the image with assigned numbers
cv2.imwrite('processed_images/numbered_rectangles.png',image)
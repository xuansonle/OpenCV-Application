from cv2 import cv2
from matplotlib import pyplot as plt
import imutils
import numpy as np
import sys

print("Preprocessing")
#Read images
img = cv2.imread("images/scan/camera2.jpg", 1)

#Convert to gray
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Blur to remove noises
blur = cv2.GaussianBlur(gray, (5, 5), 0)

#Find the edges
# _, thres = cv2.threshold(blur, 136, 200, cv2.THRESH_BINARY)
edges = cv2.Canny(blur, 50, 200, 3)

print("Contour")
#Find the contours
contours, hierarchy = cv2.findContours(
    edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

#Sort contours by size, get the largest & draw it
# largest_contour = max(contours, key = cv2.contourArea)
# largest_contour = sorted(contours, key=cv2.contourArea, reverse=True)[0:3]
# largest_contour = contours[np.argmax([cv2.contourArea(c) for c in contours])]
largest_contour = contours[np.argmax(
    [cv2.boundingRect(c)[2]*cv2.boundingRect(c)[3] for c in contours])]
# largest_contour = [contours[i] for i in np.array([cv2.boundingRect(c)[2]*cv2.boundingRect(c)[3] for c in contours]).argsort()[-10:][::-1]]

contour = img.copy()
cv2.drawContours(contour, contours, -1, (255, 0, 0), 25)

# Transform & get the output
print("Output")
try:

    #Approximate the contour
    peri = cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, 0.02 * peri, True)

    r = approx.reshape(-1, 2)  # get edges of the contour
    rect = np.zeros((4, 2), dtype='float32')

    s = np.sum(r, axis=1)
    diff = np.diff(r, axis=1)

    rect[0] = r[np.argmin(s)]  # Left up
    rect[1] = r[np.argmin(diff)]  # Right up
    rect[2] = r[np.argmax(s)]  # Right down
    rect[3] = r[np.argmax(diff)]  # Left

    # x,y-------x,y
    # |           |
    # |           |
    # x,y---------x,y

    #Get the contour width & height
    (x, y, w, h) = cv2.boundingRect(largest_contour)
    width_ = w
    height_ = h
    #Or force it to be A4 format in 300 DPI
    width_ = 1240
    height_ = 1754

    #New coordinates
    new_rect = np.array([
        [0, 0],
        [width_-1, 0],
        [width_-1, height_-1],
        [0, height_-1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, new_rect)
    output_gray = cv2.warpPerspective(gray, M, (width_, height_))
    
    output = cv2.adaptiveThreshold(output_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 2)

except Exception as e:

    print(str(e))
    sys.exit(0)

print("Print")
# MATPLOTLIB
imgs = [img, output_gray, output]
i = 1
for img in imgs:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.subplot(1, len(imgs), i)
    plt.imshow(img)
    plt.title(i)
    plt.xticks([]), plt.yticks([])
    i += 1
plt.show()

#OPENCV
# cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("Original", 1280, 720)
# cv2.imshow("Original", img)

# cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("Result", 1280, 720)
# cv2.imshow("Result", output)

# cv2.waitKey(0)
# cv2.destroyAllWindows()


# DEBUG CONTOUR
# cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("Result", 1280, 720)
# for c in largest_contour:
#     x, y, w, h = cv2.boundingRect(c)

#     cv2.rectangle(contour, (x, y), (x + w, y + h), (0, 255, 255), 5)
#     print(f"{x},{y},{w},{h}")
#     cv2.putText(contour, f"{cv2.contourArea(c)}, ({w*h})", (x, y),
#                 cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 4, (0, 0, 255), 3)

#     cv2.drawContours(contour, c, -1, (255, 0, 0), 25)
# cv2.imshow("Result", contour)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
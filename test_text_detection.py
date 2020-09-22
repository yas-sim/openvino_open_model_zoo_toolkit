import cv2
import numpy as np
import open_model_zoo_toolkit as omztk

omz = omztk.openvino_omz()
model = omz.textDetector()

cap = cv2.VideoCapture(0)

key = 0

img = cv2.imread('resources/textdet.jpg')
rects, imgs = model.run(img)
print(rects)

for rect in rects:
    box = cv2.boxPoints(rect).astype(np.int32)     # Obtain rotated rectangle
    cv2.polylines(img, [box], True, (0,255,0), 4)  # Draw bounding box

cv2.imshow('text', img)
key = cv2.waitKey(3 * 1000)     # 3 sec

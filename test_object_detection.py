import cv2
import open_model_zoo_toolkit as omztk

omz = omztk.openvino_omz()
model = omz.objectDetector()                        # Default model will be used
#model = omz.objectDetector('ssd_mobilenet_v2_coco') # Also, you can select a OMZ model from 'model_def.yml' and use it

img = cv2.imread('resources/car_1.bmp')
res = model.run(img)
print(res)
# Example: res = [[1, 'person', 0.9291767, (42, 2), (759, 714)]]

for obj in res:
    cv2.rectangle(img, obj[3], obj[4], color=(255,0,0), thickness=2)
cv2.imshow('result', img)
cv2.waitKey(3 * 1000)       # 3 seconds
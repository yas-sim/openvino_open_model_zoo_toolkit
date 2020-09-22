import cv2
import open_model_zoo_toolkit as omztk

omz = omztk.openvino_omz()
model = omz.humanPoseEstimator()

#model.setDevice('MYRIAD')

img = cv2.imread('resources/people.jpg')
res = model.run(img)
print(res)

omztk.renderPeople(img, res)
cv2.imshow('result', img)
cv2.waitKey(3 * 1000)     # 3 sec
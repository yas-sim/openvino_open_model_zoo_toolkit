import cv2
import open_model_zoo_toolkit as omztk

omz = omztk.openvino_omz()
model = omz.imageClassificator()

img = cv2.imread('resources/car.png')
res = model.run(img)
print(res)
# Example: res = [[479, 'car wheel', 0.5016654], [817, 'sports car, sport car', 0.31316656], [436, 'beach wagon, station wagon, wagon, estate car, beach waggon, station waggon, waggon', 0.06171181]]

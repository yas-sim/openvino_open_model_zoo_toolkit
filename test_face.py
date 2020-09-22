import cv2
import open_model_zoo_toolkit as omztk

omz = omztk.openvino_omz()

facedet = omz.faceDetector()
agegen  = omz.ageGenderEstimator()
hp      = omz.headPoseEstimator()
emo     = omz.emotionEstimator()
lm      = omz.faceLandmarksEstimator()

img = cv2.imread('resources/girl.jpg')
detected_faces = facedet.run(img)

for face in detected_faces:
    face_img = omztk.ocv_crop(img, face[3], face[4], scale=1.3)  # Crop detected face (x1.3 wider)
    landmarks         = lm.run(face_img)                         # Estimate facial landmark points
    # Example: landmarks = [(112, 218), (245, 192), (185, 281), (138, 369), (254, 343)]

    face_lmk_img = face_img.copy()                               # Copy cropped face image to draw markers on it
    for lmk in landmarks:
        cv2.drawMarker(face_lmk_img, lmk, (255,0,0), markerType=cv2.MARKER_TILTED_CROSS, thickness=4)  # Draw markers on landmarks
    cv2.imshow('cropped face with landmarks', face_lmk_img)
    cv2.waitKey(2 * 1000)  # 2 sec                               # Display cropped face image with landmarks

    yaw, pitch, roll = hp.run(face_img)                          # Estimate head pose (=head rotation)
    # Example: yaw, pitch, roll = -2.6668947, 22.881355, -5.5514703
    face_rot_img = omztk.ocv_rotate(face_img, roll)              # Correct roll to be upright the face

    age, gender, prob = agegen.run(face_rot_img)                 # Estimate age and gender
    print(age,gender,prob)
    # Example: age, gender, prob = 23, female, 0.8204694
    emotion           = emo.run(face_rot_img)                    # Estimate emotion
    # Example: emotion = 'smile'

    print(age, gender, emotion, landmarks)

    cv2.imshow('cropped and rotated face', face_rot_img)
    cv2.waitKey(2 * 1000)  # 2 sec

    cv2.rectangle(img, face[3], face[4], (255,0,0), 2)

cv2.imshow('result', img)
cv2.waitKey(3 * 1000)      # 3 sec

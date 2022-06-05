import cv2
import mediapipe as mp

from thin_face import thin_face
from enlarge_eyes import enlarge_eyes
from whiten_refine_face import whiten_refine_face

mp_drawing = mp.solutions.drawing_utils
mp_faces = mp.solutions.face_mesh

faces = mp_faces.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture("./src.png")
success, image = cap.read()
if success:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_height, image_width, _ = image.shape

    image.flags.writeable = False
    results = faces.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            thin_face_dst = thin_face(image, face_landmarks.landmark, image_width, image_height) # 瘦脸
            enlarge_eyes_dst = enlarge_eyes(thin_face_dst, face_landmarks.landmark, image_width, image_height) # 大眼
            whiten_refine_face_dst = whiten_refine_face(enlarge_eyes_dst) # 美白+磨皮
            cv2.imshow("src", image)
            cv2.imshow("thin_face_dst", thin_face_dst)
            cv2.imshow("enlarge_eyes_dst", image)
            whiten_refine_face_dst.show()
            whiten_refine_face_dst.save("beauty.png", quality=95)
            
    cv2.waitKey()

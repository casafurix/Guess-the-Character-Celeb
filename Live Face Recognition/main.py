# import cv2
# import face_recognition

# img1 = cv2.imread("stevecarrell.jpg")
# rgb_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
# img_encoding1 = face_recognition.face_encodings(rgb_img1)[0]

# img2 = cv2.imread("scarlettjohansson.jpg")
# rgb_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
# img_encoding2 = face_recognition.face_encodings(rgb_img2)[0]

# img3 = cv2.imread("scarlett2.jpg")
# rgb_img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
# img_encoding3 = face_recognition.face_encodings(rgb_img3)[0]

# result = face_recognition.compare_faces([img_encoding2], img_encoding3)
# print("Result: ", result)

# cv2.imshow("Img", img1)
# cv2.imshow("Img", img2)
# cv2.waitKey(0)

import cv2
from simple_facerec import SimpleFacerec

# encode faces from a folder
sfr = SimpleFacerec()
sfr.load_encoding_images("images/")

# load camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # detect faces
    face_locations, face_names = sfr.detect_known_faces(frame)
    for face_loc, name in zip(face_locations, face_names):
        y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

        cv2.putText(
            frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2
        )
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 3)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()

from cv2 import cv2

face_cascade = cv2.CascadeClassifier("xml/haarcascade_frontalface_default.xml")
# eye_cascade = cv2.CascadeClassifier("xml/haarcascade_eye.xml")

cap = cv2.VideoCapture(0)

while True:

    _, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
        
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)
        # for (sx, sy, sw, sh) in eyes:
        #     cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 255, 255), 2)

    cv2.imshow("Face Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break

cap.release()
cv2.destroyAllWindows()

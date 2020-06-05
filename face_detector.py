from cv2 import cv2

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)

while True:
    
    _, frame = cap.read()
    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
    
    for (x,y,w,h) in faces:
        
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,255,0), 2)
        
    cv2.imshow("Face Detector", frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        
        break
    
cap.release()
cv2.destroyAllWindows()
        
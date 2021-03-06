from platform import release

from cv2 import cv2

cap = cv2.VideoCapture(0)

# cap = cv2.VideoCapture("images/motion/motion.mp4")

ret1, frame1 = cap.read()
ret2, frame2 = cap.read()

img=cv2.imread("tutorial.jpg")

while True:
    
    if ret2:
    
        diff = cv2.absdiff(frame1, frame2)  #finding the absolute difference between 2 frames
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY) #convert to gray image
        blur = cv2.GaussianBlur(gray, (5,5), 0) #remove noises
        _, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY) #threshold
        dilated = cv2.dilate(thresh, None, iterations=3) #fill in the holes
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #find the contour
        
        #draw the contour:
        # cv2.drawContours(frame1, contours, -1, (0,255,255), 2) 
        for contour in contours:
            (x,y,w,h) = cv2.boundingRect(contour)
            
            if cv2.contourArea(contour) < 700:
                continue
            cv2.rectangle(frame1, (x,y), (x+w,y+h), (0,255,255), 2) #draw the box
            cv2.putText(frame1, "Status: Movement", (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255,255,0), 2)
            
        cv2.imshow("My Video", frame1)
    
        frame1 = frame2
        ret1 = ret2
        ret2, frame2 = cap.read()
            
    else:
        
        break
    
    if cv2.waitKey(1) == ord("q"):
        
        break

cap.release()
cv2.destroyAllWindows()
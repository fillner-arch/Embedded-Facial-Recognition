import cv2
import os
from authenticate_face import authenticate_face

import time
from gpiozero import LED

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

os.makedirs("authorized_faces", exist_ok=True)

red_led = LED(17)
green_led = LED(18)

def capture_face(): 

    cap = cv2.VideoCapture(0)


    while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            
            faces = face_cascade.detectMultiScale(gray,1.3,5)
            
            for (x,y,w,h) in faces:
    #             cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
                
                face_crop = frame[y:y+h, x:x+h]
                
                face_resized = cv2.resize(face_crop, (112,112))
                
                cv2.imshow("Face 112x112", face_resized)
                
            cv2.imshow("Camera", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                if len(faces) > 0:
                    filename = f"authorized_faces/face_{len(os.listdir('authorized_faces'))}.jpg"
                    cv2.imwrite(filename, face_resized)
                    print(f"Saved {filename}")
                    
            elif key == ord('a'):
                if len(faces) > 0:
                    is_authenticated, similarity = authenticate_face(face_resized)
                    print(f"Face auth = {is_authenticated} with similarity {similarity}")
                    
                    if is_authenticated:
                        
                        green_led.on()
                        red_led.off()
                        time.sleep(4)
                    else:
                        red_led.on()
                        green_led.off()
                        time.sleep(4)
            
            elif key == ord('q'):
                break
            
            else:
                red_led.off()
                green_led.off()
            
    cap.release()
    cv2.destroyAllWindows()
    
    

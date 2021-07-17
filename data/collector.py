import cv2, time
import numpy as np

###Collect 3000 images, 1000 images per each class

cam = cv2.VideoCapture(0)
n = 0
labels = ["rock", "paper", "scissors"]
index = 0
past = 0
now = 0
attempt = 0
pre = -1
print(f"Will start collecting 1000 images for class {labels[index]} after 3 seconds")
while True:
    _, frame = cam.read()
    frame = cv2.resize(frame,(448, 336))
    frame = cv2.flip(frame, 1)
    clone = frame.copy()
    cv2.rectangle(frame, (50, 100), (200, 250), (0, 255, 0), 2)
    if index != 3:
        cropped = clone[100:250, 50:200]
        cropped = cv2.resize(cropped,(64, 64))
        if attempt == 0:
            past = time.time()
            attempt = 1
        now = time.time()
        passed_time = int(now-past)
        if passed_time != pre and passed_time < 4:
            print(passed_time)
            pre = passed_time
        if passed_time >= 3:
            if n != 1000:
                n += 1
                print(f"Capturing and saving {labels[index]+str(n)}.jpg")
                cv2.imwrite(f"{labels[index]+str(n)}.jpg", cropped)
            else:
                n = 0
                past = 0
                now = 0
                attempt = 0
                pre = -1
                index += 1
                if index != 3:
                    print(f"Will start collecting 1000 images for class {labels[index]} after 3 seconds")
    else: break
    cv2.imshow("Win", frame)
    k = cv2.waitKey(30) 
    if k == ord("q"):
        break
        


cv2.destroyAllWindows()
cam.release()

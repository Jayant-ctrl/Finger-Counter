import cv2
import os 
import time
import HandTrackingModule as htm


capture = cv2.VideoCapture(0)
capture.set(3, 640)
capture.set(4, 480)

pTime = 0

folderPath = 'FingerImages'
myList = os.listdir(folderPath)
print(myList)

overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    #print(f'{folderPath}/{imPath}')
    overlayList.append(image)
print(len(overlayList))    


detector = htm.handDetector(detectionCon = 0.75)

tipIds = [4, 8, 12, 16, 20]

while True:
    isTrue, frame = capture.read()
    frame = detector.findHands(frame)
    lmList = detector.findPosition(frame, draw = False)
    #print(lmList)
    
    if len(lmList) != 0:
        
        fingers = []
        
        #Thumb
        if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
            fingers.append(1)
        else:
           fingers.append(0)
        
        # 4 Fingers
        for id in range(1,5):
         if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
            fingers.append(1)
        else:
           fingers.append(0)
        #print(fingers)
        
        #print fingers
        totalFingers = fingers.count(1)
        print(totalFingers)

        h, w, c = overlayList[totalFingers - 1].shape
        frame[0:h, 0:w] = overlayList[totalFingers - 1]
        
        cv2.rectangle(frame, (20, 225), (170, 425), (0,255,0), -1)
        cv2.putText(frame, str(totalFingers), (45, 375), cv2.FONT_HERSHEY_PLAIN, 10, (255,0,0), 25)
        
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(frame, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)

    cv2.imshow('video',frame)
    cv2.waitKey(1)    

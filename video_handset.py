import cv2 as cv
import numpy as np

capture = cv.VideoCapture(0)
index = 5
nn = 578
while True:
    ret,frame = capture.read()
    if ret is True:
        cv.imshow("frame",frame)
        index += 1
        if index % 5 == 0 :
            nn = nn+1
            cv.imwrite("D:/day_02/"+str(nn)+".jpg",frame)
        c = cv.waitKey(50)
        if c == 27:
            break
    else:
        break

cv.destroyAllWindows()
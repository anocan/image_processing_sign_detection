import numpy as np
import cv2

path = r'/Users/anilbudak/VSCode/image_processing/images/a1a3top.png'

src = cv2.imread(path, cv2.IMREAD_COLOR)
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    
def detectCircle(source,minRadius, maxRadius,CLR):
    gray = cv2.medianBlur(source, 3)
        
        
    rows = gray.shape[0]
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, rows / 16,
    param1=100, param2=30,
    minRadius=minRadius, maxRadius=maxRadius)
        
        
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv2.circle(src, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = i[2]
            cv2.circle(src, center, radius, CLR, 3)
        

    cv2.imshow("detected circles", src)
    cv2.waitKey(0)

detectCircle(gray.copy(),200,250,(255,0,0))
detectCircle(gray.copy(),260,300,(0,255,0))
import pytesseract as pyt
import cv2

path = r'/Users/anilbudak/VSCode/image_processing/images/A1.png'

img = cv2.imread(path)
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)


hImg, wImg,_ = img.shape
boxes = pyt.image_to_boxes(img,lang='eng',config='--psm 6')

for b in boxes.splitlines():
    #print(b)
    b = b.split(' ')
    print(b)

    x,y,w,h = int(b[1]),int(b[2]),int(b[3]),int(b[4])
    cv2.rectangle(img,(x,hImg-y),(w,hImg-h),(0,0,255),2)

cv2.imshow('Result',img)
cv2.waitKey(0)
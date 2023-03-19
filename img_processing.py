import numpy as np
import matplotlib.pyplot as plt
import cv2
import pytesseract as pyt

path = r'/Users/anilbudak/VSCode/image_processing/images/a1a3top.png'
COLORS = [
    (0,0,0),(255,0,0),(0,255,0),(255,0,255),(0,255,255),(255,255,0),
    (0,0,0),(255,0,0),(0,255,0),(255,0,255),(0,255,255),(255,255,0)
] 
# Contour Color Order: black, dark_blue, green, purple, yellow, light_blue

SIDES_OF_CIRCLE = 13
DETECTED = 0
PASS = 0
OUTPUT = []

img = cv2.imread(path)
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(gray_img, 127, 255, 0)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

print("Contours = {}" .format(len(contours)))


for i in range(len(contours)):
    #skip the main frame
    if i == 0:
        i=1
        continue
    
    cv2.drawContours(img, [contours[i]], 0, COLORS[i], 2) #draw contours

    approx = cv2.approxPolyDP(contours[i], 0.01 * cv2.arcLength(contours[i], True), True)
    sides = len(approx) #number of sides of the each contour

    print(len(approx))

    if (sides >= SIDES_OF_CIRCLE):
        if cv2.isContourConvex(approx):
            print('CIRCLE')
        if DETECTED:
            M1 = M
            cX1 = cX
            cY1 = cY

        M = cv2.moments(contours[i])
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        if DETECTED:
            distance = (((cX1 - cX) ** 2) + ((cY1 - cY) ** 2)) ** 0.5
            if distance<100:
                PASS = 1
                #print("pass")
                hImg, wImg,_ = img.shape
                boxes = pyt.image_to_boxes(rgb_img,lang='eng',config='--psm 6')

                for b in boxes.splitlines():
                    b = b.split(' ')
                    #print(b)
                    OUTPUT.append(b[0])
                print(OUTPUT)

            else:
                print("fail")
                #ADJUST CAMERA


        #print(cX,cY)
        cv2.putText(img, "+", (cX - 20, cY - 20),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        if PASS != 1:
            DETECTED = 1
        else:
            DETECTED = 0
    
for i in range(len(OUTPUT)):
    if OUTPUT[i] != 'l':
        continue
    else:
        OUTPUT[i] = 1
print(OUTPUT)




# Show contours on the same image in different colours   
cv2.imshow("Image with Contours", img)
cv2.waitKey()
cv2.destroyAllWindows()


"""
cv2.drawContours(img, contours, -1, (255, 0, 0) ,thickness=2)
cv2.imshow("window title", img)
cv2.waitKey()
cv2.destroyAllWindows()"""

""" Show original image
plt.figure(figsize=[5,5])
plt.imshow(rgb_img,cmap='gray') 
plt.show()

"""
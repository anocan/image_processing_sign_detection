import numpy as np
import cv2
import pytesseract as pyt
 
DEBUG = 0 #SET 1 FOR DEBUG PURPOSES
path = r'/Users/anilbudak/VSCode/image_processing/images/x2y1.png'

#Sources
src = cv2.imread(path, cv2.IMREAD_COLOR)
gray = cv2.cvtColor(src.copy(), cv2.COLOR_BGR2GRAY)

#Detection and Correction Functions
def detectCircle(source,minRadius, maxRadius,CLR): #CIRCLE DETECTION
    p0=16 #min_dist = X/16: Minimum distance between detected centers.
    p1=100 #param_1 = 200: Upper threshold for the internal Canny edge detector.
    p2=30 #param_2 = 100: Threshold for center detection.
    blurRatio = 3
    
    
    CIRCLE_COUNT=0
    CIRCLE_INFO=[]
    gray = cv2.medianBlur(source, blurRatio)
        
        
    rows = gray.shape[0]
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 
    rows / p0,
    param1=p1, param2=p2,
    minRadius=minRadius, maxRadius=maxRadius)
        
        
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            CIRCLE_COUNT += 1
            center = (i[0], i[1])
            # circle center
            cv2.circle(src, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = i[2]
            cv2.circle(src, center, radius, CLR, 3)
    else:
        if DEBUG:
            print('NO CIRCLE FOUND')
        #ACTION
        
    CIRCLE_INFO.append(CIRCLE_COUNT)
    CIRCLE_INFO.append(center)
    CIRCLE_INFO.append(radius)

    if DEBUG:
        cv2.imshow("detected circles", src)
        cv2.waitKey(0)

    return CIRCLE_INFO

def areaFilter(minArea, inputImage): #FILTER
    # Perform an area filter on the binary blobs:
    componentsNumber, labeledImage, componentStats, componentCentroids = \
        cv2.connectedComponentsWithStats(inputImage, connectivity=4)

    # Get the indices/labels of the remaining components based on the area stat
    # (skip the background component at index 0)
    remainingComponentLabels = [i for i in range(1, componentsNumber) if componentStats[i][4] >= minArea]

    # Filter the labeled pixels based on the remaining labels,
    # assign pixel intensity to 255 (uint8) for the remaining pixels
    filteredImage = np.where(np.isin(labeledImage, remainingComponentLabels) == True, 255, 0).astype('uint8')

    return filteredImage

def detectText(): #OCR
    ###APPLY COLOR FILTER FOR ACCURATE OCR

    # Conversion to CMYK (just the K channel):

    # Convert to float and divide by 255:
    imgFloat = src.astype(np.double) / 255.

    # Calculate channel K:
    kChannel = 1 - np.max(imgFloat, axis=2)

    # Convert back to uint 8:
    kChannel = (255 * kChannel).astype(np.uint8)

    binaryThresh = 190
    _, binaryImage = cv2.threshold(kChannel, binaryThresh, 255, cv2.THRESH_BINARY)
    minArea = 100
    binaryImage = areaFilter(minArea, binaryImage)

    # Use a little bit of morphology to clean the mask:
    # Set kernel (structuring element) size:
    kernelSize = 3
    # Set morph operation iterations:
    opIterations = 2
    # Get the structuring element:
    morphKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernelSize, kernelSize))
    # Perform closing:
    binaryImage = cv2.morphologyEx(binaryImage, cv2.MORPH_CLOSE, morphKernel, None, None, opIterations, cv2.BORDER_REFLECT101)

    if DEBUG:
        cv2.imshow("binaryImage [closed]", binaryImage)
        cv2.waitKey(0)
    ###

    OUTPUT = []
    CONFIG = "-c tessedit_char_whitelist=1234lABCDXYZT --psm 6"

    ###ACTUAL IMAGE TO TEXT RECOGNITION WITH DEBUG TOOLS
    boxes = pyt.image_to_boxes(binaryImage,lang='eng',config=CONFIG)

    for b in boxes.splitlines():
        b = b.split(' ')   
        if DEBUG:
            x,y,w,h = int(b[1]),int(b[2]),int(b[3]),int(b[4])
            hImg, wImg,_ = src.shape
            cv2.rectangle(src,(x,hImg-y),(w,hImg-h),(0,0,255),3)
            cv2.putText(src,b[0],(x,hImg-y+25),cv2.FONT_HERSHEY_COMPLEX,1,(50,50,255),2)
            cv2.imshow("detected circles", src)
            cv2.waitKey(0)
        OUTPUT.append(b[0])
        if DEBUG: 
            print(OUTPUT)
    ###
    return OUTPUT

def correctText(raw): #FIX MISINTERPRETATIONS (such as l->1)
    for i in range(len(raw)):
        if raw[i] != 'l':
            continue
        else:
            raw[i] = '1'
    FINAL = raw
    print(FINAL)

    return FINAL

def verifyText(input): #CHECK FOR CHARACTER LENGTH
    if len(input) != 4:
        print('MISSING CHARACTERS')
        #ADJUST CAMERA
    else:
        print('VERIFIED')
        #SEND INFO

#DETECT INNER AND OUTER CIRCLES
innerCircle=detectCircle(gray,200,250,(255,0,0))
outerCircle=detectCircle(gray,260,300,(0,255,0))
totalCircle=innerCircle[0]+outerCircle[0]

#Check for Correction and Detect Text
if totalCircle>=3:
    if totalCircle>3:
        print('100% CIRCLE VERIFICATION')
        verifyText(correctText(detectText()))
        
        #ACTION
    else:
        print('75% CIRCLE VERIFICATION')
        #ADJUST CAMERA FOR ONCE AND TRY
else:
    print('Low CIRCLE Verification')
    #ADJUST CAMERA AND TRY

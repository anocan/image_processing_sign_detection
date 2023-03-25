import numpy as np
import cv2
import pytesseract as pyt
from collections import Counter
 
DEBUG = 1 #SET 1 FOR DEBUG PURPOSES
dev_binary = 1
file_name = 'IMGPRCSNG.jpg'
cap = cv2.VideoCapture(0)
WIDTH=640
HEIGHT=480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
OUTPUT = []
THRESHOLD = 5 # HIGH VALUE = HIGH ACCURACY, MORE TIME
binaryThreshConstant = 200
verify_timer = 0
global_timer = 0
globalTimerConstant = 300

#Detection and Correction Functions

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

    global binaryThreshConstant

    # Conversion to CMYK (just the K channel):

    # Convert to float and divide by 255:
    imgFloat = frame.copy().astype(np.double) / 255.

    # Calculate channel K:
    kChannel = 1 - np.max(imgFloat, axis=2)

    # Convert back to uint 8:
    kChannel = (255 * kChannel).astype(np.uint8)

    binaryThresh = binaryThreshConstant #190
    _, binaryImage = cv2.threshold(kChannel, binaryThresh, 255, cv2.THRESH_BINARY)
    minArea = 80 #100
    binaryImage = areaFilter(minArea, binaryImage)

    # Use a little bit of morphology to clean the mask:
    # Set kernel (structuring element) size:
    kernelSize = 3
    # Set morph operation iterations:
    opIterations = 2 #2
    # Get the structuring element:
    morphKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernelSize, kernelSize))
    # Perform closing:
    binaryImage = cv2.morphologyEx(binaryImage, cv2.MORPH_CLOSE, morphKernel, None, None, opIterations, cv2.BORDER_REFLECT101)

    if dev_binary:
        cv2.imshow('binaryFrame', binaryImage)

    OUTPUT = []
    CONFIG = "-c tessedit_char_whitelist=1234lABCDXYZT --psm 6"

    ###ACTUAL IMAGE TO TEXT RECOGNITION WITH DEBUG TOOLS
    boxes = pyt.image_to_boxes(binaryImage,lang='eng',config=CONFIG)

    for b in boxes.splitlines():
        b = b.split(' ')   
        if DEBUG:
            x,y,w,h = int(b[1]),int(b[2]),int(b[3]),int(b[4])
            hImg, wImg,_ = frame.shape
            cv2.rectangle(frame,(x,hImg-y),(w,hImg-h),(0,0,255),3)
            cv2.putText(frame,b[0],(x,hImg-y+25),cv2.FONT_HERSHEY_COMPLEX,1,(50,50,255),2)
        OUTPUT.append(b[0])
        #if DEBUG: 
        #    print(OUTPUT)
    ###
    return OUTPUT

def verifyText(input): #CHECK FOR CHARACTER LENGTH
    global verify_timer
    global binaryThreshConstant
    timer_threshold = 20 
    binaryThreshRate = 5
    #0.25

    if len(input) != 4:
        if verify_timer<timer_threshold:
            verify_timer += 1
        elif verify_timer>=timer_threshold:
            if binaryThreshConstant != 150:
                binaryThreshConstant -= binaryThreshRate
                verify_timer = 0
                print(binaryThreshConstant)
    else:
        if input[2] == '2':
            input[2] = 'Z'
        for i in range(len(input)):
            if input[i] != 'l':
                continue
            else:
                input[i] = '1'
        OUTPUT.append(input)

def eliminate(i):
    accepted_first = {'A','B','C','D'}
    accepted_second = {'1','2','3','4'}
    accepted_third = {'X','Y','Z','T'}
    accepted_fourth = {'1','2','3'}

    if i[0] not in accepted_first:
        return 0
    elif i[1] not in accepted_second:
        return 0
    elif i[2] not in accepted_third:
        return 0
    elif i[3] not in accepted_fourth:
        return 0
    else:
        return 1

def similarity(i):
    global global_timer
    global binaryThreshConstant

    if not i:
        global_timer = 0
        binaryThreshConstant = 200
        return 0
    out = []
    for l in range(4):
        temp = []
        for k in i:
            temp.append(k[l])
        out.append(max(z for z,v in Counter(temp).items() if v>1))
    return out

####MAIN FUNCTION

if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    global_timer += 1

    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here

    cv2.imwrite(file_name,frame)
    gray = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)

    #Check for Correction and Detect Text
    verifyText(detectText())
            
    if DEBUG and global_timer>=globalTimerConstant:
        print(global_timer)

    if len(OUTPUT)>=THRESHOLD or global_timer>=globalTimerConstant:
        c=0
        for i in OUTPUT:
            if eliminate(i) == 0:
                c=1
                OUTPUT.remove(i)
                #print(i)
        if c == 0 or global_timer>=globalTimerConstant:
            global_timer=0
            print(similarity(OUTPUT))
        #print(OUTPUT)
            #ACTION

    # Display the resulting frame
    if not dev_binary:
        cv2.imshow('frame', frame)


    if cv2.waitKey(1) == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
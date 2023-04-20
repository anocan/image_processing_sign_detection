import numpy as np
import cv2
import pytesseract as pyt
from collections import Counter
import math
import itertools
import serial
import os
import time
 
DEBUG = 1 #SET 1 FOR DEBUG PURPOSES
dev_binary = 0 #1 FOR BINARY FRAME 0 FOR OCR FRAME
file_name = 'IMGPRCSNG.jpg'
data_path = 'data.txt'
WIDTH=480
HEIGHT=360

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

OUTPUT = [] #SINGLE SIGN
RESULT = [] #ALL OF THE SIGNS
THRESHOLD = 5 # HIGH VALUE = HIGH ACCURACY, MORE TIME
signCount = 0
binaryThreshConstant = 210 #200
binaryLowerConstant = 180
verify_timer = 0
global_timer = 0
globalTimerConstant = 40 # seconds
globalCropCoX = 6
globalCropCoY = 8

move = "move"
ocr = 100
path = '/dev/cu.usbserial-110' #/dev/ttyACM0 , /dev/cu.usbserial-110

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
    global globalCropCoX
    global globalCropCoY

    # Conversion to CMYK (just the K channel):
    binaryFrame = frame.copy()
    originX=0
    originY=0
    cropX=math.floor(WIDTH/globalCropCoX)
    cropY=math.floor(HEIGHT/globalCropCoY)
    crop_binary = binaryFrame[originY+cropY:originY+(HEIGHT-cropY), originX+cropX:originX+(WIDTH-cropX)]
    crop_frame = frame[originY+cropY:originY+(HEIGHT-cropY), originX+cropX:originX+(WIDTH-cropX)]

    # Convert to float and divide by 255:
    imgFloat = crop_binary.astype(np.double) / 255.

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
            hImg, wImg,_ = crop_frame.shape
            cv2.rectangle(crop_frame,(x,hImg-y),(w,hImg-h),(0,0,255),3)
            cv2.putText(crop_frame,b[0],(x,hImg-y+25),cv2.FONT_HERSHEY_COMPLEX,1,(50,50,255),2)
        OUTPUT.append(b[0])
        #if DEBUG: 
        #    print(OUTPUT)
    ###
    return OUTPUT

def verifyText(input): #CHECK FOR CHARACTER LENGTH
    global verify_timer
    global binaryThreshConstant
    global binaryLowerConstant
    timer_threshold = 10 
    binaryThreshRate = 5
    #0.5

    if len(input) != 4:
        if verify_timer<timer_threshold:
            verify_timer += 1
        elif verify_timer>=timer_threshold:
            if binaryThreshConstant != binaryLowerConstant:
                binaryThreshConstant -= binaryThreshRate
                verify_timer = 0
                print(binaryThreshConstant)
            else:
                binaryThreshConstant = 210
    else:
        if input[2] == '2':
            input[2] = 'Z'
        if input[1] == 'A':
            input[1] = '4'
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

    out = []
    for l in range(4):
        temp = []
        for k in i:
            temp.append(k[l])
        out.append(max((z for z,v in Counter(temp).items() if v>1),default='NULL'))
        if 'NULL' in out:
            global_timer = 0
            binaryThreshConstant = 210
            OUTPUT.clear()
    return out

def nullBreaker(input):
    rm=[]
    for i in range(len(input)):
        for r in range(4):
            if input[i][r] == 'NULL':
                rm.append(input[i])
    l2 = [x for x in input if x not in rm]
    return l2

def bluntAlgoConverter(input):
    consec = list(itertools.chain.from_iterable(input))
    l3 = []
    for i in range(0,len(consec),2):
        l3.append(consec[i]+consec[i+1])
    return l3

####MAIN FUNCTION

if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    start_time = time.time()
    
    #CHECK IF ARDUINO AT THE LAST SIGN
    #IF NOT
        #TRIGGER ARDUINO TO MOVE 
        #WAIT COMMAND FROM ARDUINO -> signCount++
        #TRIGGER OCR
        #GO BACK TO THE ALGORITHM
    #IF YES
        #if 'NULL' in RESULT: RESULT.remove('NULL') ['D','4','NULL','3']
        #CONVERT DATA ELIGIBLE TO BLUNTALGO 
        #WRITE AT A TXT FILE
        #CALL BLUNTALGO SCRIPT
        #EXIT PROGRAM

    #if signCount != 8:
        ##### TRIGGER ARDUINO TO MOVE & WAIT COMMAND FROM ARDUINO
        #if __name__ == '__main__':
            #ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
            #ser.reset_input_buffer()

            #while True:
                #ser.write('move'.encode('utf-8')) #SEND 'move' TO ARDUINO
                #response = ser.read()
                #if response != b'':
                    #if int.from_bytes(response, byteorder='big') == 200:
                        ##print('stopped')
                        #break
    #########
    if signCount != 8:
        if __name__ == '__main__':
            ser = serial.Serial(path, 9600, timeout=1)
            ser.reset_input_buffer()

        while True:
            ser.write(move.encode('utf-8'))
            response = ser.read()
            print('Response= {}' . format(response))
            if response != b'':
                if response[0] == ocr:
                    # DO OCR
                    print('moved!')
                    ser.close()
                    break
            time.sleep(1)

        while True:
            end_time = time.time()
            global_timer = end_time - start_time

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
                print('OVERTIME')

            if len(OUTPUT)>=THRESHOLD or global_timer>=globalTimerConstant:
                c=0
                for i in OUTPUT:
                    if eliminate(i) == 0:
                        c=1
                        OUTPUT.remove(i)
                        #print(i)
                if c == 0 or global_timer>=globalTimerConstant:
                    global_timer=0
                    binaryThreshConstant=210
                    #print(similarity(OUTPUT))

                    ###ACTION
                    RESULT.append(similarity(OUTPUT))
                    print(RESULT)
                    OUTPUT.clear()
                    signCount+=1
                    break

            # Display the resulting frame
            if DEBUG and not dev_binary:
                originX=0
                originY=0
                cropX=math.floor(WIDTH/globalCropCoX)
                cropY=math.floor(HEIGHT/globalCropCoY)
                cv2.imshow('frame', frame[originY+cropY:originY+(HEIGHT-cropY), originX+cropX:originX+(WIDTH-cropX)])


            if cv2.waitKey(1) == ord('q'):
                break
    ######### 

    else:
        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()
        ### GO BACK TO START POINT

        ###
        RESULT = nullBreaker(RESULT)
        RESULT = bluntAlgoConverter(RESULT) #Convert Data Eligible to bluntAlgo
        with open(data_path, 'w') as f:
            f.write(str(RESULT))
        break

#os.system('bluntAlgo.py')
#exit()
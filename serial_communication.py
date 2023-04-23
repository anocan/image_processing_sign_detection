import serial
import time
from datetime import datetime

move = "move"
ocr = 100
brk = 45
path = '/dev/cu.usbserial-110' #/dev/ttyACM0 , /dev/cu.usbserial-110
signCount = 0
totalSigns = 8

def serialCom(mode):
    if __name__ == '__main__':
        ser = serial.Serial(path, 9600, timeout=1)
        ser.reset_input_buffer()

        while True:
            t = 0.01
            if not (mode == "only-read"):
                ser.write(move.encode('utf-8'))
                response = ser.read()
                print('{}:{} Writing {}, Waiting for {}... Response= {}' . format(datetime.now().minute,datetime.now().second,move,brk,response))
                if response != b'':
                    if response[0] == brk:
                        # loop broken
                        print('loop broken')
                        ser.close()
                        break
                time.sleep(t)
            else:
                response = ser.read()
                print('{}:{} Waiting for {}... Response= {}' . format(datetime.now().minute,datetime.now().second,ocr,response))
                if response != b'':
                    if response[0] == ocr:
                        # DO OCR
                        print('ocr')
                        ser.close()
                        break
                time.sleep(t)

while not (signCount == totalSigns):
    serialCom("only-read")
    print("PROGRESS")
    serialCom("write-read")
    print("-------------- {}" .format(signCount))
    signCount += 1
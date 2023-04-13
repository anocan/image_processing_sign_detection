import serial

com = ['A1', 'X2', 'D1', 'Y1']
path = '/dev/cu.usbserial-110' #/dev/ttyACM0

if __name__ == '__main__':
    ser = serial.Serial(path, 9600, timeout=1)
    ser.reset_input_buffer()

    while True:
        #ser.write(100)
        ser.write(str(com).encode('utf-8')) #Serial.readString()
        response = ser.read()
        if response != b'':
            if int.from_bytes(response, byteorder='big') == 200:
                print('stop')
                break
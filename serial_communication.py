import serial

if __name__ == '__main__':
    ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
    ser.reset_input_buffer()

    while True:
        ser.write('move'.encode('utf-8'))
        response = ser.read()
        if response != b'':
            if int.from_bytes(response, byteorder='big') == 200:
                print('stop')
                break
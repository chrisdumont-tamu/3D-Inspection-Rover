#this code will test the basic navigation under the new grid assumption
#under this assumption all tomato plants are 4ft away from each other.
#the rover needs to travel exactly this far before stopping 
#after stopping, daltons system will begin it's work

import serial   #allows reading and writing via serial comms
import time     #allows the time library to be used

#setup of serial functionality
port = "/dev/ttyUSB0"  #assigns the serial port to USB 0 on the raspberry pi"
ser = serial.Serial()
ser.port = port
ser.close() 
ser.baudrate = 9600
ser.bytesize = serial.EIGHTBITS
ser.parity = serial.PARITY_NONE
ser.stopbits = serial.STOPBITS_ONE
ser.timeout = 1
ser.open()
ser.flushInput()
ser.flushOutput()

def Stop():
    ser.write(bytes[0]) #stops all rover motors

def Travel_Forward_4ft():
    print ("Traveling forward 4ft")
    ser.write(bytes[70])
    ser.write(bytes[198])
    time.sleep(5)

Travel_Forward_4ft()
Stop()
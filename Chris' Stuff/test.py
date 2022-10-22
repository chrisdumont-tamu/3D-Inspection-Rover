#this code will test the basic navigation under the new grid assumption
#under this assumption all tomato plants are 4ft away from each other.
#the rover needs to travel exactly this far before stopping 
#after stopping, daltons system will begin it's work


import serial
import time

#setup of serial functionality
port = "/dev/ttyUSB0"  #find available USB using "ls /dev/*USB* in terminal"
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

#v=d/t
#t=d/v

v=9 # roughly 9 cm/seconds

def Stop():
  ser.write(bytes([0]))

def HowFar():
  distance=float(input("How far do I need to go? (cm please) : "))
  t=distance/v
  t=round(t,2)
  print("traveling forward for " ,t, " seconds")
  ser.write(bytes([70]))
  ser.write(bytes([198]))
  time.sleep(t)

HowFar()
Stop()

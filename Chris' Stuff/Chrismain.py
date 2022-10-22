########################################################################
# THINGS TO KNOW
        # MOTOR INFO
# Sample for Sabertooth Simplified Serial
# Test w/ RPi3B+ -> USB to TTL 5V Cable -> Sabertooth 2X12
# USB to TTL Cable Hookup to Sabertooth: GND -> 0V, TX -> S1
# DIP Config (9600 Buad): 1UP - 2DOWN - 3 UP - 4DOWN - 5UP - 6UP

# M1: 1 - 127; 1 = full reverse, 64 = stop, 127 = full forward
# M2: 128 - 255; 128 = full reverse, 192 = stop, 255 = full forward
# M1 & M2; 0 = stop
#stop +/- 6 is good for slow movement
#Avoid going full speed

        # DISTANCE SENSOR INFO
#SENSOR 3
#Trigger=16
#Echo=18
#SENSOR 2
#Trigger=36
#Echo=32
#SENSOR 1
#Trigger=8
#Echo=38
########################################################################

# import all needed libraries
# serial for motor controls
# time for motors and sensors
# GPIO for distance sensors

import serial
import time
import RPi.GPIO as GPIO

#setup serial functionality
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

#Initialize aisle count
Current_Aisle=0

#setup RPi.GPIO functionality
# uses RPi board numbers and ignores initial states
GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)

#define constants
#CONSTANTS FOR DISTANCE SENSORS
  # trigger pins
Trigger1 = 8
Trigger2=36
Trigger3=16
  # echo pins
Echo1 = 38
Echo2=32
Echo3=18

# movement constants
v=3.54  # cm/second
pause=1
avoid_distance=100 #needs testing, how far the avoidance codes travel
avoid_time=3.5 # needs testing, how long the avoidance code takes

#setup GPIO functionality
  # output pins
GPIO.setup(Trigger1, GPIO.OUT)
GPIO.setup(Trigger2,GPIO.OUT)
GPIO.setup(Trigger3, GPIO.OUT)
  # input pins
GPIO.setup(Echo1, GPIO.IN)
GPIO.setup(Echo2,GPIO.IN)
GPIO.setup(Echo3,GPIO.IN)

print("All setups and constants declared successfully \n")
time.sleep(1)

# DISTANCE SENSOR CODES
print("I will now test all distance sensors \n", )

# distance sensor 1 complete cycle
def Distance_Sensor_1():
  GPIO.output(Trigger1, GPIO.LOW)
  time.sleep(0.5)
  GPIO.output(Trigger1, GPIO.HIGH)
  time.sleep(0.00001)
  GPIO.output(Trigger1, GPIO.LOW)
  while GPIO.input(Echo1)==0:
    pulse_start_time = time.time()
  while GPIO.input(Echo1)==1:
    pulse_end_time = time.time()

  pulse_duration = pulse_end_time - pulse_start_time
  distance1 = round(pulse_duration * 17150, 2)
  print ("Distance sensor 1 reads:",distance1,"cm")
  
  return distance1

# distance sensor 2 complete cycle
def Distance_Sensor_2():
  GPIO.output(Trigger2, GPIO.LOW)
  time.sleep(0.5)
  GPIO.output(Trigger2, GPIO.HIGH)
  time.sleep(0.00001)
  GPIO.output(Trigger2, GPIO.LOW)
  while GPIO.input(Echo2)==0:
    pulse_start_time = time.time()
  while GPIO.input(Echo2)==1:
    pulse_end_time = time.time()

  pulse_duration = pulse_end_time - pulse_start_time
  distance2 = round(pulse_duration * 17150, 2)
  print ("Distance sensor 2 reads:",distance2,"cm")
  
  return distance2

# distance sensor 3 complete cycle  
def Distance_Sensor_3():
  GPIO.output(Trigger3, GPIO.LOW)
  time.sleep(0.5)
  GPIO.output(Trigger3, GPIO.HIGH)
  time.sleep(0.00001)
  GPIO.output(Trigger3, GPIO.LOW)
  while GPIO.input(Echo3)==0:
    pulse_start_time = time.time()
  while GPIO.input(Echo3)==1:
    pulse_end_time = time.time()

  pulse_duration = pulse_end_time - pulse_start_time
  distance3 = round(pulse_duration * 17150, 2)
  print ("Distance sensor 3 reads:",distance3,"cm \n")

  return distance3
  
# test of distance sensors
Distance_Sensor_1()
Distance_Sensor_2()
Distance_Sensor_3()
print("All distance sensors successful \n")
time.sleep(0.5)

#MOTOR CODES

#test for functionality code

# Bring all motors to a stop
def Stop():
  ser.write(bytes([0]))
  
# spins left and right forward for 1 second 
def Forward_Test():
  print("traveling forward")
  ser.write(bytes([70]))
  ser.write(bytes([198]))
  time.sleep(pause)
  Stop()
  time.sleep(pause)

# spins left and right backward for 1 second
def Backward_Test():
  print("reversing")
  ser.write(bytes([58]))
  ser.write(bytes([186]))
  time.sleep(pause)
  Stop()
  time.sleep(pause)

# Turns the rover right 60 degrees by:
# reversing right motors and forward left motor
# NEEDS TWEAKING
def Turn_Right_60():
  print("turning right")
  ser.write(bytes([40]))
  ser.write(bytes([214]))
  time.sleep(0.65)
  Stop()
  time.sleep(pause)

# Turns the rover right 60 degrees by:
# reversing left motors and forward right motor
# NEEDS TWEAKING
def Turn_Left_60():
  print("turning left")
  ser.write(bytes([80]))
  ser.write(bytes([160]))
  time.sleep(0.65)
  Stop()
  time.sleep(pause)

#  test functionality of above 4 codes  
print("I will now test motor controls \n")
time.sleep(0.5)
Forward_Test()
Turn_Left_60()
Turn_Right_60()
Backward_Test()
print("All motor controls successful \n")
time.sleep(pause)

print ("All Test successful \n")

#Here I will begin defining my true codes
# spins both motors forward indefinetly
def Forward():
  print("traveling forward")
  ser.write(bytes([70]))
  ser.write(bytes([198]))

# takes user input to know distance forward needed to travel
# uses input to calculate travel time needed
# returns this value and the input value
def How_Far():
  plant_distance=float(input("How far do I need to go? (cm please)\n and no value greater than 120 : "))
  travel_time=plant_distance/v  
  return plant_distance, travel_time

#Avoidance Codes

# has rover move to the left around an object
def Avoid_Left():
  Turn_Left_60()
  Stop()
  Forward()
  time.sleep(pause*1.5)
  Turn_Right_60()
  Forward()
  time.sleep(pause*2)
  Stop()
  Turn_Right_60()
  Stop()
  Forward()
  time.sleep(pause*1.5)
  Turn_Left_60()
  Stop()

# has rover move to the left around an object
def Avoid_Right():
  Turn_Right_60()
  Stop()
  Forward()
  time.sleep(pause*1.5)
  Turn_Left_60()
  Forward()
  time.sleep(pause*2)
  Stop()
  Turn_Left_60()
  Stop()
  Forward()
  time.sleep(pause*1.5)
  Turn_Right_60()
  Stop()

# has the rover turn right very slightly
def Turn_Right_10():
  print("Correcting course right")
  ser.write(bytes([40]))
  ser.write(bytes([214]))
  time.sleep(0.2*pause)
  Stop()
  time.sleep(pause)

# has the rover turn right very slightly
def Turn_Left_10():
  print("Correcting course left")
  ser.write(bytes([80]))
  ser.write(bytes([160]))
  time.sleep(0.2*pause)
  Stop()
  time.sleep(pause)

# gets rover close to object in path
# NEEDS TESTING
def Approach_Obstacle(distance1,distance2,distance3):
    avg_distance = (distance1+distance2+distance3)/3
    avg_distance = round(avg_distance,2)
    approach_distance = avg_distance-10
    approach_time = approach_distance/v
    Forward()
    pause(approach_time)
    Stop()

# Hear is the the whole enchilada
def main():
  # asks the user how far the rover needs to travel  
  plant_distance,travel_time=How_Far()
  # initiates infinte for loop
  while True:
      # saves all distance readings to variable
    distance1=Distance_Sensor_1()
    distance2=Distance_Sensor_2()
    distance3=Distance_Sensor_3()

    # dispays all pertinent data
    print("values brought in are as follows:\n")
    print("distances from sensors=\n",distance1, "\n", distance2, "\n" ,distance3)
    print("travel distance= " , plant_distance)
    print("calculated travel time= ", travel_time, "\n")

    #Scenario: Object to large to go around
    if ((distance1 < plant_distance) and (distance2 < plant_distance) and (distance3 < plant_distance)):
      print("getting closer to object")  
      Approach_Obstacle(distance1,distance2,distance3)
      print ("Danger Close\n")
      Stop()
      obstacle_moved=int(input("Has the object been moved? (yes=1 or no=2) : "))
      while(obstacle_moved != 1):
        obstacle_moved=int(input("Has the object been moved? (yes=1 or no=2) : "))
    
    # Scenario: Objet takes up right side of path
    elif ((distance1 < plant_distance) and (distance2 < plant_distance) and (distance3 > plant_distance)):
      if (plant_distance < avoid_distance):
        print("getting closer to object")  
        Approach_Obstacle(distance1,distance2,distance3)
        print("object is too close to plant")
        Stop()
        break
      else:
        print("getting closer to object")
        Approach_Obstacle(distance1,distance2,distance3)
        print("Avoid Left\n")
        Avoid_Left()
        plant_distance = plant_distance - avoid_distance
        travel_time = travel_time - avoid_time
    
    # Scenario: Object takes up left side of path
    elif ((distance1 > plant_distance) and (distance2 < plant_distance) and (distance3 < plant_distance)):
      if (plant_distance < avoid_distance):
        print("getting closer to object")  
        Approach_Obstacle(distance1,distance2,distance3)
        print("object is too close to plant")
        Stop()
        break
      else:
        print("getting closer to object")  
        Approach_Obstacle(distance1,distance2,distance3)  
        print("Avoid Right\n")
        Avoid_Right()
        plant_distance = plant_distance - avoid_distance
        travel_time = travel_time - avoid_time

    # Scenario: Narrow object in middle of path
    elif ((distance1 > plant_distance) and (distance2 < plant_distance) and (distance3 > plant_distance)):
      if (plant_distance < avoid_distance):
        print("getting closer to object")  
        Approach_Obstacle(distance1,distance2,distance3)  
        print("object is too close to plant")
        Stop()
        break
      else:
        print("getting closer to object")  
        Approach_Obstacle(distance1,distance2,distance3)  
        print("Small Object dead ahead\nAvoiding Left\n")
        Avoid_Left()
        plant_distance = plant_distance - avoid_distance
        travel_time = travel_time - avoid_time
    
    # Scenario: Small object on left side of path
    elif ((distance1 < plant_distance) and (distance2 > plant_distance) and (distance3 > plant_distance)):
      print("I need to turn left a little bit")
      Turn_Left_10()
    
    # Scenario: Small object of right side of path
    elif ((distance1 > plant_distance) and (distance2 > plant_distance) and (distance3 < plant_distance)):
      print("I need to turn right a little bit")
      Turn_Right_10()

    # Scenario: No object in path
    else:
      print("No Danger Detected\n")
      Forward()
      time.sleep(travel_time)
      Stop()
      main()
      
# runs it    
print("All other functions declare successfully\n\n")
main()
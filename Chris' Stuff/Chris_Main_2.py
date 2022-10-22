import time
import RPi.GPIO as GPIO
import serial
import Chris_Functions as CF

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

def main():
  # asks the user how far the rover needs to travel  
  plant_distance,travel_time=How_Far()
  # initiates infinte for loop
  while True:
      # saves all distance readings to variable
    distance1=CF.Distance_Sensor_1()
    distance2=CF.Distance_Sensor_2()
    distance3=CF.Distance_Sensor_3()

    # dispays all pertinent data
    print("values brought in are as follows:\n")
    print("distances from sensors=\n",distance1, "\n", distance2, "\n" ,distance3)
    print("travel distance= " , plant_distance)
    print("calculated travel time= ", travel_time, "\n")

    #Scenario: Object to large to go around
    if ((distance1 < plant_distance) and (distance2 < plant_distance) and (distance3 < plant_distance)):
      print("getting closer to object")  
      CF.Approach_Obstacle(distance1,distance2,distance3)
      print ("Danger Close\n")
      CF.Stop()
      obstacle_moved=int(input("Has the object been moved? (yes=1 or no=2) : "))
      while(obstacle_moved != 1):
        obstacle_moved=int(input("Has the object been moved? (yes=1 or no=2) : "))
    
    # Scenario: Objet takes up right side of path
    elif ((distance1 < plant_distance) and (distance2 < plant_distance) and (distance3 > plant_distance)):
      if (plant_distance < avoid_distance):
        print("getting closer to object")  
        CF.Approach_Obstacle(distance1,distance2,distance3)
        print("object is too close to plant")
        CF.Stop()
        break
      else:
        print("getting closer to object")
        CF.Approach_Obstacle(distance1,distance2,distance3)
        print("Avoid Left\n")
        CF.Avoid_Left()
        plant_distance = plant_distance - avoid_distance
        travel_time = travel_time - avoid_time
    
    # Scenario: Object takes up left side of path
    elif ((distance1 > plant_distance) and (distance2 < plant_distance) and (distance3 < plant_distance)):
      if (plant_distance < avoid_distance):
        print("getting closer to object")  
        CF.Approach_Obstacle(distance1,distance2,distance3)
        print("object is too close to plant")
        CF.Stop()
        break
      else:
        print("getting closer to object")  
        CF.Approach_Obstacle(distance1,distance2,distance3)  
        print("Avoid Right\n")
        CF.Avoid_Right()
        plant_distance = plant_distance - avoid_distance
        travel_time = travel_time - avoid_time

    # Scenario: Narrow object in middle of path
    elif ((distance1 > plant_distance) and (distance2 < plant_distance) and (distance3 > plant_distance)):
      if (plant_distance < avoid_distance):
        print("getting closer to object")  
        CF.Approach_Obstacle(distance1,distance2,distance3)  
        print("object is too close to plant")
        CF.Stop()
        break
      else:
        print("getting closer to object")  
        CF.Approach_Obstacle(distance1,distance2,distance3)  
        print("Small Object dead ahead\nAvoiding Left\n")
        CF.Avoid_Left()
        plant_distance = plant_distance - avoid_distance
        travel_time = travel_time - avoid_time
    
    # Scenario: Small object on left side of path
    elif ((distance1 < plant_distance) and (distance2 > plant_distance) and (distance3 > plant_distance)):
      print("I need to turn left a little bit")
      CF.Turn_Left_10()
    
    # Scenario: Small object of right side of path
    elif ((distance1 > plant_distance) and (distance2 > plant_distance) and (distance3 < plant_distance)):
      print("I need to turn right a little bit")
      CF.Turn_Right_10()

    # Scenario: No object in path
    else:
      print("No Danger Detected\n")
      CF.Forward_()
      time.sleep(travel_time)
      CF.Stop()
      main()
      
# runs it    
print("All other functions declare successfully\n\n")
main()
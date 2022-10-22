import time
import RPi.GPIO as GPIO
import serial

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

#Main code for rover navigation
def Main_Chris():
    #Determine Constants
    plant_distance=121.92
    distance1=Distance_Sensor_1()
    distance2=Distance_Sensor_2()
    distance3=Distance_Sensor_3()

    #Display Pertinent Data
    print("distances from sensors=\n",distance1, "\n", distance2, "\n" ,distance3)
    print("travel distance= " , plant_distance)

    #Scenario 1: Distance too large to go around
    if ((distance1 < plant_distance) and (distance2 < plant_distance) and (distance3 < plant_distance)):
        print("getting closer to object")  
        Approach_Obstacle(distance1,distance2,distance3)
        print ("Danger Close\n")
        Stop()
        obstacle_moved=int(input("Has the object been moved? (yes=1 or no=2) : "))
        while(obstacle_moved != 1):
            obstacle_moved=int(input("Has the object been moved? (yes=1 or no=2) : "))

    #Scenario 2: Object takes up right side of path
    # Scenario: Objet takes up right side of path
    elif ((distance1 < plant_distance) and (distance2 < plant_distance) and (distance3 > plant_distance)):
      if (plant_distance < avoid_distance):
        print("getting closer to object")  
        Approach_Obstacle(distance1,distance2,distance3)
        print("object is too close to plant")
        Stop()
      else:
        print("getting closer to object")
        Approach_Obstacle(distance1,distance2,distance3)
        print("Avoid Left\n")
        Avoid_Left()
        plant_distance = plant_distance - avoid_distance
        Forward()
        time.sleep(plant_distance/v)
        Stop()
    
    #Scenario 3: Object takes up left side of path
    elif ((distance1 > plant_distance) and (distance2 < plant_distance) and (distance3 < plant_distance)):
      if (plant_distance < avoid_distance):
        print("getting closer to object")  
        Approach_Obstacle(distance1,distance2,distance3)
        print("object is too close to plant")
        Stop()
      else:
        print("getting closer to object")  
        Approach_Obstacle(distance1,distance2,distance3)  
        print("Avoid Right\n")
        Avoid_Right()
        plant_distance = plant_distance - avoid_distance
        Forward()
        time.sleep(plant_distance/v)
        Stop()

    #Scenario 4: Narrow object in middle of path
    elif ((distance1 > plant_distance) and (distance2 < plant_distance) and (distance3 > plant_distance)):
      if (plant_distance < avoid_distance):
        print("getting closer to object")  
        Approach_Obstacle(distance1,distance2,distance3)  
        print("object is too close to plant")
        Stop()
      else:
        print("getting closer to object")  
        Approach_Obstacle(distance1,distance2,distance3)  
        print("Small Object dead ahead\nAvoiding Left\n")
        Avoid_Left()
        plant_distance = plant_distance - avoid_distance
        Forward()
        time.sleep(plant_distance/v)
        Stop()

    #Scenario 5: Small object on left side of path
    elif ((distance1 < plant_distance) and (distance2 > plant_distance) and (distance3 > plant_distance)):
        print("I need to turn left a little bit")
        Turn_Left_10()

    #Scenario 6: Small object of right side of path
    elif ((distance1 > plant_distance) and (distance2 > plant_distance) and (distance3 < plant_distance)):
        print("I need to turn right a little bit")
        Turn_Right_10()

    #Scenario: No Object in path
    else:
        print("No Danger Detected\n")
        Travel_Forward_4ft()
        Stop()
#This is all of my functions outside of main
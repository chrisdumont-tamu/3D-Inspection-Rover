###############################
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
#########################################################

#general imports for code functionality

import serial
import time
from time import sleep
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

GPIO.setwarnings(False) #sets all gpio warnings to 0
GPIO.setmode(GPIO.BOARD) #sets gpio numbering system to match board numbers

#Initialize Aisle Count
Aisle_Count=0

#Define Constants
#Distance Sensors
  # trigger pins
Trigger1 = 8
Trigger2=36
Trigger3=16
  # echo pins
Echo1 = 38
Echo2=32
Echo3=18
#^^^^^^^^ Need to change GPIO 12/32 to different pins
v=10
avoid_distance=100

#Servo Controls
pan_s = 12 #GPIO for pan servo
tilt_s = 32 #GPIO for tilt servo
f_pwm = 50 #50Hz signal

# Setup for Pan/Tilt Servos
def setup():#setup function
    global pwm #pwm global
    GPIO.setup(pan_s, GPIO.OUT)#setup pan output pin
    GPIO.setup(tilt_s, GPIO.OUT) #setup tilt output pin
    GPIO.setup(15,GPIO.OUT) # enable, was 22
    GPIO.setup(13,GPIO.OUT) # direction, was 21
    GPIO.setup(11,GPIO.OUT) # step, was 20

# Setup Distance Sensors
  # output pins
GPIO.setup(Trigger1, GPIO.OUT)
GPIO.setup(Trigger2,GPIO.OUT)
GPIO.setup(Trigger3, GPIO.OUT)
  # input pins
GPIO.setup(Echo1, GPIO.IN)
GPIO.setup(Echo2,GPIO.IN)
GPIO.setup(Echo3,GPIO.IN)

print("All setups and constants declared successfully \n")

#Defines PWM Frequence for Pan/Tilt Servos
def setServoAngle(servo, angle):
    pwm = GPIO.PWM(servo, f_pwm)#setup output frequency
    pwm.start(8)
    dutyCycle = angle / 18. + 2.
    pwm.ChangeDutyCycle(dutyCycle)
    time.sleep(1.5)

#Daltons Code for Servo Control
def Main_Dalton():
    setup()  
    GPIO.output(15,GPIO.LOW)#enable turned on
    delay = 0.0000000001 #delay for the created pwm
    GPIO.output(13,GPIO.LOW) #direction chosen: HIGH=CCW, LOW=CW

    print("Engage Hyper Drive")#start
    tic = int(time.perf_counter())#timer start
    tac = 0.0#total elapsed time initialized
    while tac < 5: #loop to run for 24 seconds
        GPIO.output(11,GPIO.HIGH) #step off
        sleep(delay)
        GPIO.output(11, GPIO.LOW)#step on
        sleep(delay)
        toc = int(time.perf_counter())#timer end
        tac = toc - tic #total time elapsed 
        #print(tac)
    print(tac)
    print("That's no moon...")
    confid_rating = int(input("Enter desired confidence rating: "))#input of desired conf rating
    random_conf = 0 # initializing random  confidence variable
    i = 0 #direction identifier
    j = 0 #counter for loop iterations
    #while loop for running through directions
    #tic = time.perf_counter()
    setServoAngle(pan_s,76)#baseline pan angle
    setServoAngle(tilt_s,90)#baseline tilt angle
    while j < 1:
        #setup base position
        #setServoAngle(pan_s,76)
        #setServoAngle(tilt_s,90)
        f = open('tilt_up_test.txt')
        setServoAngle(pan_s,91)#pan right
        i = 1
        random_conf = int(f.readline())#random number testing confidence
        if random_conf > confid_rating:
            f.close()
            break
        setServoAngle(pan_s,76)#recenter
        setServoAngle(tilt_s,75)#tilt down
        i = 2
        random_conf = int(f.readline())#random number testing confidence
        if random_conf > confid_rating:
            f.close()
            break
        setServoAngle(tilt_s,90)#Recenter
        setServoAngle(pan_s,61)#pan left
        i = 3
        random_conf = int(f.readline())#random number testing confidence
        if random_conf > confid_rating:
            f.close()
            break
        setServoAngle(pan_s,76)#recenter
        setServoAngle(tilt_s,105)#tilt up
        i = 4
        random_conf = int(f.readline())#random number testing confidence
        if random_conf > confid_rating:
            f.close()
            break
        #Recenter
        j += 1
        #break
    if j ==1 :
        print("Confidence is at maximum: ", confid_rating)
    elif i == 1:
        print("Pan right increased confidence with a value of: ", random_conf)
    elif i == 2:
        print("Tilt down increased confidence with a value of: ", random_conf)
    elif i == 3:
        print("Pan left increased confidecne with a value of: ", random_conf)
    elif i == 4:
        print("Tilt up increased confidence with a value of: ", random_conf)
        #print(toc-tic)
    else:
        print("Max confidence reached")
    #run 2 
    print("Engage Hyper Drive")#start
    tic = int(time.perf_counter())#timer start
    tac = 0.0#total elapsed time initialized
    while tac < 5: #loop to run for 24 seconds
        GPIO.output(11, GPIO.HIGH) #step off
        sleep(delay)
        GPIO.output(11, GPIO.LOW)#step on
        sleep(delay)
        toc = int(time.perf_counter())#timer end
        tac = toc - tic #total time elapsed 
        #print(tac)
    print(tac)
    print("That's no moon...")
    confid_rating = int(input("Enter desired confidence rating: "))#input of desired conf rating
    random_conf = 0 # initializing random  confidence variable
    i = 0 #direction identifier
    j = 0 #counter for loop iterations
    #while loop for running through directions
    #tic = time.perf_counter()
    setServoAngle(pan_s,76)#baseline pan angle
    setServoAngle(tilt_s,90)#baseline tilt angle
    while j < 1:
        #setup base position
        #setServoAngle(pan_s,76)
        #setServoAngle(tilt_s,90)
        f = open('tilt_down_test.txt')
        setServoAngle(pan_s,91)#pan right
        i = 1
        random_conf = int(f.readline())#random number testing confidence
        if random_conf > confid_rating:
            f.close()
            break
        setServoAngle(pan_s,76)#recenter
        setServoAngle(tilt_s,75)#tilt down
        i = 2
        random_conf = int(f.readline())#random number testing confidence
        if random_conf > confid_rating:
            f.close()
            break
        setServoAngle(tilt_s,90)#Recenter
        setServoAngle(pan_s,61)#pan left
        i = 3
        random_conf = int(f.readline())#random number testing confidence
        if random_conf > confid_rating:
            f.close()
            break
        setServoAngle(pan_s,76)#recenter
        setServoAngle(tilt_s,105)#tilt up
        i = 4
        random_conf = int(f.readline())#random number testing confidence
        if random_conf > confid_rating:
            f.close()
            break
        #Recenter
        j += 1
        #break
    if j ==1 :
        print("Confidence is at maximum: ", confid_rating)
    elif i == 1:
        print("Pan right increased confidence with a value of: ", random_conf)
    elif i == 2:
        print("Tilt down increased confidence with a value of: ", random_conf)
    elif i == 3:
        print("Pan left increased confidecne with a value of: ", random_conf)
    elif i == 4:
        print("Tilt up increased confidence with a value of: ", random_conf)
        #print(toc-tic)
    else:
        print("Max confidence reached")
    
    #run 3
    print("Engage Hyper Drive")#start
    tic = int(time.perf_counter())#timer start
    tac = 0.0#total elapsed time initialized
    while tac < 5: #loop to run for 24 seconds
        GPIO.output(11, GPIO.HIGH) #step off
        sleep(delay)
        GPIO.output(11, GPIO.LOW)#step on
        sleep(delay)
        toc = int(time.perf_counter())#timer end
        tac = toc - tic #total time elapsed 
    #print(tac)
    print(tac)
    print("That's no moon...")
    confid_rating = int(input("Enter desired confidence rating: "))#input of desired conf rating
    random_conf = 0 # initializing random  confidence variable
    i = 0 #direction identifier
    j = 0 #counter for loop iterations
    #while loop for running through directions
    #tic = time.perf_counter()
    setServoAngle(pan_s,76)#baseline pan angle
    setServoAngle(tilt_s,90)#baseline tilt angle
    while j < 1:
        #setup base position
        #setServoAngle(pan_s,76)
        #setServoAngle(tilt_s,90)
        f = open('pan_right_test.txt')
        setServoAngle(pan_s,91)#pan right
        i = 1
        random_conf = int(f.readline())#random number testing confidence
        if random_conf > confid_rating:
            f.close()
            break
        setServoAngle(pan_s,76)#recenter
        setServoAngle(tilt_s,75)#tilt down
        i = 2
        random_conf = int(f.readline())#random number testing confidence
        if random_conf > confid_rating:
            f.close()
            break
        setServoAngle(tilt_s,90)#Recenter
        setServoAngle(pan_s,61)#pan left
        i = 3
        random_conf = int(f.readline())#random number testing confidence
        if random_conf > confid_rating:
            f.close()
            break
        setServoAngle(pan_s,76)#recenter
        setServoAngle(tilt_s,105)#tilt up
        i = 4
        random_conf = int(f.readline())#random number testing confidence
        if random_conf > confid_rating:
            f.close()
            break
        #Recenter
        j += 1
        #break
    if j ==1 :
        print("Confidence is at maximum: ", confid_rating)
    elif i == 1:
        print("Pan right increased confidence with a value of: ", random_conf)
    elif i == 2:
        print("Tilt down increased confidence with a value of: ", random_conf)
    elif i == 3:
        print("Pan left increased confidecne with a value of: ", random_conf)
    elif i == 4:
        print("Tilt up increased confidence with a value of: ", random_conf)
        #print(toc-tic)
    else:
        print("Max confidence reached")
    #run 4
    print("Engage Hyper Drive")#start
    tic = int(time.perf_counter())#timer start
    tac = 0.0#total elapsed time initialized
    while tac < 5: #loop to run for 24 seconds
        GPIO.output(11, GPIO.HIGH) #step off
        sleep(delay)
        GPIO.output(11, GPIO.LOW)#step on
        sleep(delay)
        toc = int(time.perf_counter())#timer end
        tac = toc - tic #total time elapsed 
    #print(tac)
    print(tac)
    print("That's no moon...")
    confid_rating = int(input("Enter desired confidence rating: "))#input of desired conf rating
    random_conf = 0 # initializing random  confidence variable
    i = 0 #direction identifier
    j = 0 #counter for loop iterations
    #while loop for running through directions
    #tic = time.perf_counter()
    setServoAngle(pan_s,76)#baseline pan angle
    setServoAngle(tilt_s,90)#baseline tilt angle
    while j < 1:
        #setup base position
        #setServoAngle(pan_s,76)
        #setServoAngle(tilt_s,90)
        f = open('pan_left_test.txt')
        setServoAngle(pan_s,91)#pan right
        i = 1
        random_conf = int(f.readline())#random number testing confidence
        if random_conf > confid_rating:
            f.close()
            break
        setServoAngle(pan_s,76)#recenter
        setServoAngle(tilt_s,75)#tilt down
        i = 2
        random_conf = int(f.readline())#random number testing confidence
        if random_conf > confid_rating:
            f.close()
            break
        setServoAngle(tilt_s,90)#Recenter
        setServoAngle(pan_s,61)#pan left
        i = 3
        random_conf = int(f.readline())#random number testing confidence
        if random_conf > confid_rating:
            f.close()
            break
        setServoAngle(pan_s,76)#recenter
        setServoAngle(tilt_s,105)#tilt up
        i = 4
        random_conf = int(f.readline())#random number testing confidence
        if random_conf > confid_rating:
            f.close()
            break
        #Recenter
        j += 1
        #break
    if j ==1 :
        print("Confidence is at maximum: ", confid_rating)
    elif i == 1:
        print("Pan right increased confidence with a value of: ", random_conf)
    elif i == 2:
        print("Tilt down increased confidence with a value of: ", random_conf)
    elif i == 3:
        print("Pan left increased confidecne with a value of: ", random_conf)
    elif i == 4:
        print("Tilt up increased confidence with a value of: ", random_conf)
        #print(toc-tic)
    else:
        print("Max confidence reached")
    #reset

    GPIO.output(15,GPIO.LOW)#enable turned on
    delay = 0.0000000001 #delay for the created pwm
    GPIO.output(13,GPIO.HIGH) #direction chosen: HIGH=CCW, LOW=CW

    print("Engage Hyper Drive")#start
    tic = int(time.perf_counter())#timer start
    tac = 0.0#total elapsed time initialized
    while tac < 20.0: #loop to run for 24 seconds
        GPIO.output(11, GPIO.HIGH) #step off
        sleep(delay)
        GPIO.output(11, GPIO.LOW)#step on
        sleep(delay)
        toc = int(time.perf_counter())#timer end
        tac = toc - tic #total time elapsed 
        #print(tac)
    
    print(tac)
    print("Great shot kid, that's one in a million!")

#Distance Sensor Code and Reading
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

#Motor Controls: Rover
# Bring all motors to a stop
def Stop():
    ser.write(bytes([0]))

def Turn_Right_60():
    print("turning right")
    ser.write(bytes([40]))
    ser.write(bytes([214]))
    time.sleep(0.65)
    Stop()
    time.sleep(pause)

def Turn_Left_60():
    print("turning left")
    ser.write(bytes([80]))
    ser.write(bytes([160]))
    time.sleep(0.65)
    Stop()
    time.sleep(pause)

def Forward():
    print("traveling forward")
    ser.write(bytes([70]))
    ser.write(bytes([198]))

#Rover Avoidance Codes
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

# has the rover go forward 4ft
def Travel_Forward_4ft():
    print ("Traveling forward 4ft")
    ser.write(bytes[70])
    ser.write(bytes[198])
    time.sleep(5)

#has rover get close to obstacle before avoiding
def Approach_Obstacle(distance1,distance2,distance3):
    avg_distance = (distance1+distance2+distance3)/3
    avg_distance = round(avg_distance,2)
    approach_distance = avg_distance-10
    drive_time = approach_distance/v
    Forward()
    time.sleep(drive_time)
    Stop()

# Turn rover right 90 degrees
def Turn_90():
    print("Turning Right 90 Degrees")
    ser.write(bytes([40]))
    ser.write(bytes([214]))
    time.sleep(1.8)
    Stop()

#Rover Travels to next aisle
def Travel_to_next_Aisle():
    Travel_Forward_4ft()
    Turn_90()
    Travel_Forward_4ft()
    Turn_90()
    Travel_Forward_4ft()
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
    
def The_Whole_Enchilada():
    Plant_Number=0
    while(Plant_Number<4):
        Main_Chris()
        Main_Dalton()
        Plant_Number+=1
        print("You are currently on plant # ", Plant_Number)
    # After traveling the row, in this case 4 plants, the rover will navigate to the next aisle over
    Aisle_Count+=1
    print("You are now on Ailse # ", Aisle_Count)
    if Aisle_Count==3: #if on the last aisle [3] the code finishes
        print("Finished with greenhouse! WHOOP")
    else:
        Plant_Number=0
        Travel_to_next_Aisle()
        The_Whole_Enchilada()


The_Whole_Enchilada()

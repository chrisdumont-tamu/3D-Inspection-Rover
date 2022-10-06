# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 22:17:02 2022

@author: dalto
"""

import RPi.GPIO as GPIO
import time
#import random 

pan_s = 12 #GPIO for pan servo
tilt_s = 32 #GPIO for tilt servo
f_pwm = 50 #50Hz signal
GPIO.setwarnings(False)


#start functions
def setup():#setup function
    global pwm #pwm global
    GPIO.setmode(GPIO.BOARD) #board setup
    GPIO.setup(pan_s, GPIO.OUT)#setup pan output pin
    GPIO.setup(tilt_s, GPIO.OUT) #setup tilt output pin
    

def setServoAngle(servo, angle):
  pwm = GPIO.PWM(servo, f_pwm)#setup output frequency
  pwm.start(8)#pwm start
  dutyCycle = angle / 18. + 2. #duty cycle calc
  pwm.ChangeDutyCycle(dutyCycle)
  time.sleep(1.5)#wait time
#end functions

print("starting")
setup()#run setup for pi
setServoAngle(pan_s,76)#set pan angle
setServoAngle(tilt_s,90)#set tilt angle
f = open('tilt_up_test.txt','r')#read file with confidence values
confid_rating = int(input("Enter desired confidence rating: "))#input of desired conf rating
random_conf = 0 # initializing random  confidence variable that is read from file
i = 0 #direction identifier
j = 0 #counter for loop iterations
max_c = 0 #max confidence
moved = []#directions moved
pan_center = 76 #pan reallignment angle
tilt_center = 90 #tilt reallignment angle
#while loop for running through directions
while j < 1:
    while j < 1:
        #setup base position
        #setServoAngle(pan_s,76)
        #setServoAngle(tilt_s,90)
        setServoAngle(pan_s,(pan_center + 10))#pan right
        i = 1#designator for right
        random_conf = int(f.readline())#random number testing confidence
        if random_conf > confid_rating: #testing confidence against desired value
            max_c = random_conf #sets max confidence value
            moved.insert(0, i)#inserts direction into list
            break #break loops if used
        setServoAngle(pan_s, pan_center)#recenter
        setServoAngle(tilt_s,(tilt_center-10))#tilt down
        i = 2 #designator for down
        random_conf = int(f.readline())#random number testing confidence
        if random_conf > confid_rating:
            max_c = random_conf
            moved.insert(0, i)
            break
        setServoAngle(tilt_s,tilt_center)#Recenter
        setServoAngle(pan_s,(pan_center-10))#pan left
        i = 3 #designator for left
        random_conf = int(f.readline())#random number testing confidence
        if random_conf > confid_rating:
            max_c = random_conf
            moved.insert(0, i)
            break
        setServoAngle(pan_s,pan_center)#recenter
        setServoAngle(tilt_s,(tilt_center+10))#tilt up
        i = 4 #designator for up
        random_conf = int(f.readline())#random number testing confidence
        if random_conf > confid_rating:
            max_c = random_conf
            moved.insert(0, i)
            break
    #Recenter
        j += 1
    #break

#Reset Gimbal Positioning
setServoAngle(pan_s, 76)
setServoAngle(tilt_s, 90)

#print statements for 
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
else:
    print("Max confidence reached")

#pwm = GPIO.PWM(servo, f_pwm)
#pwm = GPIO.PWM(servo, f_pwm)
#pwm.stop()
GPIO.cleanup()
print(moved)
f.close()
print("Done")
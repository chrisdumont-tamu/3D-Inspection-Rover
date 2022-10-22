# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 13:10:54 2022

@author: dalto
"""
import time
#run 1
print("Engage Hyper Drive")#start
tic = int(time.perf_counter())#timer start
tac = 0.0#total elapsed time initialized
while tac < 60: #loop to run for 24 seconds
  '''GPIO.output(11, GPIO.HIGH) #step off
  sleep(delay)
  GPIO.output(11, GPIO.LOW)#step on
  sleep(delay)'''
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
while j < 1:
    f = open('tilt_down_test.txt','r')
    #pan right
    i = 1
    random_conf = int(f.readline())#random number testing confidence
    if random_conf > confid_rating:
        f.close()
        break
    #Recener
    #tilt down
    i = 2
    random_conf = int(f.readline())#random number testing confidence
    if random_conf > confid_rating:
        f.close()
        break
    #Recenter
    #pan left
    i = 3
    random_conf = int(f.readline())#random number testing confidence
    if random_conf > confid_rating:
        f.close()
        break
    #Recenter
    #tilt up
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
while tac < 60: #loop to run for 24 seconds
  '''GPIO.output(11, GPIO.HIGH) #step off
  sleep(delay)
  GPIO.output(11, GPIO.LOW)#step on
  sleep(delay)'''
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
while j < 1:
    f = open('tilt_up_test.txt','r')
    #pan right
    i = 1
    random_conf = int(f.readline())#random number testing confidence
    if random_conf > confid_rating:
        f.close()
        break
    #Recener
    #tilt down
    i = 2
    random_conf = int(f.readline())#random number testing confidence
    if random_conf > confid_rating:
        f.close()
        break
    #Recenter
    #pan left
    i = 3
    random_conf = int(f.readline())#random number testing confidence
    if random_conf > confid_rating:
        f.close()
        break
    #Recenter
    #tilt up
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
while tac < 60: #loop to run for 24 seconds
  '''GPIO.output(11, GPIO.HIGH) #step off
  sleep(delay)
  GPIO.output(11, GPIO.LOW)#step on
  sleep(delay)'''
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
while j < 1:
    f = open('pan_right_test.txt','r')
    #pan right
    i = 1
    random_conf = int(f.readline())#random number testing confidence
    if random_conf > confid_rating:
        f.close()
        break
    #Recener
    #tilt down
    i = 2
    random_conf = int(f.readline())#random number testing confidence
    if random_conf > confid_rating:
        f.close()
        break
    #Recenter
    #pan left
    i = 3
    random_conf = int(f.readline())#random number testing confidence
    if random_conf > confid_rating:
        f.close()
        break
    #Recenter
    #tilt up
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
while tac < 60: #loop to run for 24 seconds
  '''GPIO.output(11, GPIO.HIGH) #step off
  sleep(delay)
  GPIO.output(11, GPIO.LOW)#step on
  sleep(delay)'''
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
while j < 1:
    f = open('pan_left_test.txt','r')
    #pan right
    i = 1
    random_conf = int(f.readline())#random number testing confidence
    if random_conf > confid_rating:
        f.close()
        break
    #Recener
    #tilt down
    i = 2
    random_conf = int(f.readline())#random number testing confidence
    if random_conf > confid_rating:
        f.close()
        break
    #Recenter
    #pan left
    i = 3
    random_conf = int(f.readline())#random number testing confidence
    if random_conf > confid_rating:
        f.close()
        break
    #Recenter
    #tilt up
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
'''
GPIO.output(15,GPIO.LOW)#enable turned on
delay = 0.0000000001 #delay for the created pwm
GPIO.output(13,GPIO.HIGH) #direction chosen: HIGH=CCW, LOW=CW

print("Engage Hyper Drive")#start
tic = int(time.perf_counter())#timer start
tac = 0.0#total elapsed time initialized
while tac < 240.0: #loop to run for 24 seconds
  GPIO.output(11, GPIO.HIGH) #step off
  sleep(delay)
  GPIO.output(11, GPIO.LOW)#step on
  sleep(delay)
  toc = int(time.perf_counter())#timer end
  tac = toc - tic #total time elapsed 
  #print(tac)
  
print(tac)
print("That's no moon...")
GPIO.cleanup()
'''
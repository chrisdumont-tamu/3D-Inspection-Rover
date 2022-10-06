# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 22:12:35 2022

@author: dalto
"""
#code will test confidence in random directions and take action once increased confidence is found
import random

tensorInput = input("Input Confidence Rating: ") #input from the ML
currConfidence = 0 #variable for current confidence
#variables for holding if direction has been used or not
Lefter = 1 #number 1
Righter = 2 #number 2
Upper = 3 #number 3
Downer = 4 #number 4
Optimal_conf = 60 #variable for optimal confidence
rand_list = [1,2,3,4] #list for random to parce through
moved_list = [] #list for moved directions
#while loop to test confidence of new movements

while Optimal_conf < currConfidence:
    #mv_length = length(moved_list) #length of moved directions
    rand_num = random.choice(rand_list)#pick random numebr from list
    for i in moved_list: #goes through list to see if direction has been used
        if i != rand_num: #if i doesnt match random direction chosen
            if rand_num == 1:
                print("move left")
            elif rand_num == 2:
                print("move right")
            elif rand_num == 3:
                print("move up")
            elif rand_num == 4:
                print("move down")
            
            #print("Direction not used")
        else:
            rand_num = random.choice(rand_list)
    
                
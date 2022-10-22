# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 21:09:04 2022

@author: dalto
"""
import random #random module
#functions

#end of functions
rand_list = [1,2,3,4] #list of direction numbers
moved_list = [4,1] #list of directiuons used
counter = 0 #counter
total_nums = [] #Total directions tried
lng = len(moved_list)
#rand_num = random.choice(rand_list)
"""
for i in moved_list:
    if i != rand_num:
        print("Direction not used")
    else:
        print("direction used")
"""
rand_num = random.choice(rand_list)#pick random numebr from list
total_nums.insert(0, rand_num)#add direction to list
if lng != 0:#if length of moved list not zero
    if lng == 3:
        for i in moved_list: #goes through list to see if direction has been used
            if moved_list[0] != rand_num and moved_list[1] != rand_num and moved_list[2] != rand_num: #if i doesnt match random direction chosen
                if rand_num == 1:#move left check
                    print("move left")
                    moved_list.insert(0, rand_num)
                    break
                elif rand_num == 2:#move right check
                    print("move right")
                    moved_list.insert(0, rand_num)
                    break
                elif rand_num == 3:#move up check
                    print("move up")
                    moved_list.insert(0, rand_num)
                    break
                elif rand_num == 4:#move down check
                    print("move down")
                    moved_list.insert(0, rand_num)
                    break
    elif lng == 2:
        for i in moved_list: #goes through list to see if direction has been used
            if moved_list[0] != rand_num and moved_list[1] != rand_num: #if i doesnt match random direction chosen
                if rand_num == 1:#move left check
                    print("move left")
                    moved_list.insert(0, rand_num)
                    break
                elif rand_num == 2:#move right check
                    print("move right")
                    moved_list.insert(0, rand_num)
                    break
                elif rand_num == 3:#move up check
                    print("move up")
                    moved_list.insert(0, rand_num)
                    break
                elif rand_num == 4:#move down check
                    print("move down")
                    moved_list.insert(0, rand_num)
                    break
    elif lng == 1:
        for i in moved_list: #goes through list to see if direction has been used
            if moved_list[0] != rand_num: #if i doesnt match random direction chosen
                if rand_num == 1:#move left check
                    print("move left")
                    moved_list.insert(0, rand_num)
                    break
                elif rand_num == 2:#move right check
                    print("move right")
                    moved_list.insert(0, rand_num)
                    break
                elif rand_num == 3:#move up check
                    print("move up")
                    moved_list.insert(0, rand_num)
                    break
                elif rand_num == 4:#move down check
                    print("move down")
                    moved_list.insert(0, rand_num)
                    break
            #print("Direction not used")
            else:#direction is in moved list
                rand_num = random.choice(rand_list)#new random number chosen
                total_nums.insert(0, rand_num)#add to direction list
                counter +=1#counter up 1
else:#if moved list length is zero
        if rand_num == 1:#left check
            print("move left")
            moved_list.insert(0, rand_num)
        elif rand_num == 2:#right check
            print("move right")
            moved_list.insert(0, rand_num)
        elif rand_num == 3:#up check
            print("move up")
            moved_list.insert(0, rand_num)
        elif rand_num == 4:#down check
            print("move down")
            moved_list.insert(0, rand_num)

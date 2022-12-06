This is the 3D Inspection Rover read me file:
Dr. Jang is our professor, Dr. Nowka is our sponser, and Eric Robles is our TA.
Our team is team #8. Team members are: Christopher Dumont, Dalton Hines, Felipe Villegas, and Celeste Waters.
Please look in team updates folder for each of our weekly updates and our individual folders for any and all updated codes.

Project Setup and Running
- Move the 3D Inspection Rover to an open space, any obstructions in front of the rover should be removable
- Connect the rover's raspberry pi to a portable power source and power it as indicated by an active red LED
- Power up the rover motors by flipping the switch that is located on the rover frame
- Connect to the raspberry pi by SSH to access the linux terminal.
    - To access the use the command 'ssh -Y pi@raspberrypi.local'
    - The password is 'raspberry'
- Use the 'cd BOB_Inspection_Rover' linux command to navigate from the home directory to the 'BOB_Inspection_Rover' folder which contains the codes to run the project
- Use the command 'python BLS.py' to begin the run
    - The file 'BLS.py' is the main integration code used








Mast and Gimbal Code Details:
-Has necessary setups within functions for proper running
-Angle and runtime can be changed for gimbal/mast movement
-Number of stops per plant can be changed
-Takes 4 minutes for mast to start from bottom to reach top, divide 240 seconds by the number of stops wanted to determine runtime between stops
-Make sure the downward movement runtime matches the upward motion runtime
-Angles are in degrees and time is in seconds
-Make sure mast driver is conneted to battery

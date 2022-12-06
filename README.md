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


Navigation Code Details:
- All functions and set ups are in the file Chris_Functions
- this code can also be used to test all functionality of the navigation components
- Current set up is for a plant to be 8 feet away
-   this can be altered in the beginning of the code where costants are declared
- all distances used by the navigation code are in centimeters
-   withing code there is detailed comments in the begging detailing how to plug in all the sensors and driver connection to a raspberry pi 4b
- there is also comments one every functions detailing how it works.

Mast and Gimbal Code Details:
-Has necessary setups within functions for proper running
-Angle and runtime can be changed for gimbal/mast movement
-Number of stops per plant can be changed
-Takes 4 minutes for mast to start from bottom to reach top, divide 240 seconds by the number of stops wanted to determine runtime between stops
-Make sure the downward movement runtime matches the upward motion runtime
-Angles are in degrees and time is in seconds
-Make sure mast driver is conneted to battery

######################################### EE2703 Applied Programming Lab - Jan-May 2021########################################
###########################
######### Name          : Pnn Sumedh
######### Roll.No       : EE19B047
######### Assignment.No : 1
######### Commandline   : python ee19b047_file1.py <path to netlist file>
###########################

import sys
import os.path                                                          
from os import path

filepath = sys.argv[1]                                                                          # Accessing file path from command argument            

if path.exists(filepath) == False:                                                              # Checking for file in the given path
    print("Error! File with the given file name doesnot exist in the given path.")
    exit()                                                                                      # Exiting the program if the file is not present

f = open(filepath)                                                                              # Opening the file using file handling
line = f.readlines()                                                                            # Reading the file line by line
f.close()                                                                                       # Closing the file

index_circuit = None
index_end = None
for x in line:
    if ".circuit" == x[:len(".circuit")]:                                                       # checking for ".circuit" in each line
	    if index_circuit == None:                                                               # if the .circuit detected is the first one then index of .circuit is stored
	        index_circuit = line.index(x)
	    else:                                                                                   # if the .circuit detected is not the first one then the netlist file is Malformed and the the program is exited
	        print("Invalid!! Malformed input file!!")
	        exit()
    elif ".end" == x[:len(".end")]:                                                             # checking for ".end" in each line
	    if index_end == None:                                                                   # if the .end detected is the first one then index of .circuit is stored
	        index_end = line.index(x)
	    else:                                                                                   # if the .end detected is not the first one then the netlist file is Malformed and the the program is exited
	        print("Invalid!! Malformed input file!!")
	        exit()

if type(index_end) != int or type(index_circuit) != int or index_circuit>index_end:             # Checking for errors in the input file 
                                                                                                # 1) if .end is not found in the file
                                                                                                # 2) if .circuit is not found in the file  
                                                                                                # 3) if .end is detected first and then .circuit is detected
    print("Invalid!! Malformed input file!!")
    exit()

circuit = line[(index_circuit+1):index_end][::-1]                                               # Extracting the lines inbetween .circuit and .end

circuit_f = []
for x in circuit:                                                       
    indx = x.find('#')                                                                          # Checking for comments in the circuit definition
    if indx != -1:                                                                              # if there are comments in some line of the circuit definition, the part only before comments is taken
        x = x[0:indx]
    temp = x.split()                                                                            # Splitting the string with space as the division character     
    temp = ' '.join(temp[::-1])                                                                 # Joining the elements with space as the division
    temp = temp.replace('\n','')                                                                # Removing \n characters if there are any
    if temp != '':
         circuit_f.append(temp)                                                                 # Storing the circuit definition lines in a new list

print("\nCircuit :")                                                                            
for x in circuit_f:                                                                             # Printing the circuit definition after reversing
    print(x)
    
########################### End Of Code #############################
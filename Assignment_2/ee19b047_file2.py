######################################### EE2703 Applied Programming Lab - Jan-May 2021########################################
###########################
######### Name          : Pnn Sumedh
######### Roll.No       : EE19B047
######### Assignment.No : 2
######### Commandline   : python ee19b047_file2.py <path to netlist file>
###########################
from cmath import exp, pi, polar
import numpy as n
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

L = False                                                                                       # Checking for presence of Inductor
AC = False                                                                                      # Checking if the circuit contains AC or not
index_ac = None
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
    elif ".ac" == x[:len(".ac")]:
        AC = True
        index_ac = line.index(x)                                                                # Storing the index of line which contains the .ac command

if type(index_end) != int or type(index_circuit) != int or index_circuit>index_end:             # Checking for errors in the input file 
                                                                                                # 1) if .end is not found in the file
                                                                                                # 2) if .circuit is not found in the file  
                                                                                                # 3) if .end is detected first and then .circuit is detected
    print("Invalid!! Malformed input file!!")
    exit()
if AC: 
    w = 2*pi*float(line[index_ac].split()[-1])                                               # Storing the angular frequency if there is AC source in the circuit
else:
    w = 2*pi*float(1e-10)

circuit = line[(index_circuit+1):index_end][::-1]                                               # Extracting the lines inbetween .circuit and .end
V_size = 0                                                                                      # To count the number of voltage sources
N_size = 1                                                                                      # To count the number of nodes in the circuit
voltages ={}                                                                                    # Dictionary to store all the voltage sources
nodes = {'0':'GND'}                                                                             # Dictionary to store all the nodes in the circuit
circuit_f = []
for x in circuit:                                                       
    indx = x.find('#')                                                                          # Checking for comments in the circuit definition
    if indx != -1:                                                                              # if there are comments in some line of the circuit definition, the part only before comments is taken
        x = x[0:indx]
    temp = x.split()                                                                            # Splitting the string with space as the division character     
    try:
        if temp[0][0] == 'V':
            voltages['V'+str(V_size)] = temp[0]                                                     # Storing the name of voltage source in the dictionary
            temp[0] = 'V'+str(V_size)
            V_size += 1
    except IndexError:
            continue
        
        
    for i in [1,2]:
        if temp[i] not in nodes.values() and temp[i] != 'GND':                                  # Checking whether the node is already stored in the dictionary
            nodes[str(N_size)] = temp[i]                                                        # Storing the node in dictionary if it is not present in the dictionary
            temp[i] = str(N_size)
            N_size += 1
        elif temp[i] in nodes.values():                                                         # if the node is already in dictionary, then just renameing the node according to the dictionary
            temp[i] = list(nodes.keys())[list(nodes.values()).index(temp[i])]
        elif temp[i] == 'GND':                                                                  # if the node is Ground then renaming it as 0
            temp[i] = '0'
            
    temp = ' '.join(temp)                                                                       # Joining the elements with space as the division
    temp = temp.replace('\n','')                                                                # Removing \n characters if there are any
    if temp != '':
         circuit_f.append(temp)                                                                 # Storing the circuit definition lines in a new list

A_size = V_size + N_size


class Component:                                                                                # Creating a class named component
# Inputs to the class = Circuit definition line of a component of type string

    def __init__(self, compoinfo):
        self.compo = compoinfo.split()
        self.name = self.compo[0]                                                               # Storing the name of component in self.name
        if AC == False:                                                                         
            self.value = float(self.compo[-1])                                                  # Storing the value of the component if there is no AC source
            self.phase = 0.0
        elif AC and (self.compo[0][0] == 'I' or self.compo[0][0] == 'V') :
            self.value = float(self.compo[-2])/float(2)                                                # If there is AC source and the component is either Current source or Voltage source
            self.phase = float(self.compo[-1])*pi/180                                                  # Then storing both the value and the phase of the components
        elif AC and (self.compo[0][0] != 'I' or self.compo[0][0] != 'V'):
            self.value = float(self.compo[-1])                                                  # If there is AC source, then storing the value of components other than the sources
        self.node1 = self.compo[1]                                                              # Storing the nodes between which the component is connected
        self.node2 = self.compo[2]                                                              # Storing the nodes between which the component is connected

    
    instamp = n.zeros((A_size,A_size),n.complex)                                                # Creating a numpy array to give the conductance stamp of the component
    outstamp = n.zeros((A_size,1), n.complex)                                                   # Creating a numpy array to give the input stamp of the component
    def stamp(self):                                                                            # defining a function to create a stamp
        self.instamp = n.zeros((A_size,A_size),dtype = complex)
        self.outstamp = n.zeros((A_size,1),dtype = complex)
        
        # Creating stamp of Resistor
        
        if self.compo[0][0] == 'R':
            self.instamp[int(self.node1)][int(self.node1)] = 1/self.value
            self.instamp[int(self.node1)][int(self.node2)] = -1/self.value
            self.instamp[int(self.node2)][int(self.node1)] = -1/self.value
            self.instamp[int(self.node2)][int(self.node2)] = 1/self.value
            return self.instamp, self.outstamp
        
        # Creating stamp of Voltage source
        # The convention for Voltage source taken is Vn1 - Vn2 = Value and the direction of Current through the voltage source is from n1 to n2
        elif self.compo[0][0] == 'V':
            self.instamp[N_size+int(self.name[1:])][int(self.node1)] = 1
            self.instamp[N_size+int(self.name[1:])][int(self.node2)] = -1
            self.outstamp[N_size+int(self.name[1:])][0] = self.value*exp(1j*self.phase)     # If there is an AC source, then the outstamp will be a complex
            self.instamp[int(self.node1)][N_size+int(self.name[1:])] = -1
            self.instamp[int(self.node2)][N_size+int(self.name[1:])] = 1
            return self.instamp, self.outstamp
        
        # Creating stamp of Current source
        # Convention for Current source taken is Current flows from node n1 to node n2
        elif self.compo[0][0] == 'I':
            self.outstamp[int(self.node1[1:])][0] = -self.value*exp(1j*self.phase)          # If there is AC source, then the outstamp will be complex and the phase phasor must be multiplied to the value
            self.outstamp[int(self.node2[1:])][0] = self.value*exp(1j*self.phase)
            return self.instamp, self.outstamp

        # Creatinng stamp for Inductor
        elif self.compo[0][0] == 'L':
            self.instamp[int(self.node1)][int(self.node1)] = -1j/(self.value*w)
            self.instamp[int(self.node1)][int(self.node2)] = 1j/(self.value*w)
            self.instamp[int(self.node2)][int(self.node1)] = 1j/(self.value*w)
            self.instamp[int(self.node2)][int(self.node2)] = -1j/(self.value*w)
            return self.instamp, self.outstamp
           
        
        # Creating the stamp of Capacitor
        
        elif self.compo[0][0] == 'C':
            self.instamp[int(self.node1)][int(self.node1)] = 1j*(self.value*w)
            self.instamp[int(self.node1)][int(self.node2)] = -1j*(self.value*w)
            self.instamp[int(self.node2)][int(self.node1)] = -1j*(self.value*w)
            self.instamp[int(self.node2)][int(self.node2)] = 1j*(self.value*w)
             
            return self.instamp, self.outstamp

comp_size = 0   
components = {}
M = n.zeros((A_size,A_size),dtype = complex)                                                    # Defining the conductance matrix of the circuit
b = n.zeros((A_size,1),dtype = complex)                                                         # Defining the source matrix of the circuit

for i in range(len(circuit_f)):                                                                 
    components['Cp'+str(i)] = Component(circuit_f[i])                                           
    InStamp, OutStamp = components['Cp'+str(i)].stamp()
    M += InStamp                                                                                # Updating the conductance matrix
    b += OutStamp                                                                               # Updating the Source Matrix

M = M[1:,1:]                                                                                    # Removing the ground node row and column
b = b[1:]
Ans_rect = n.linalg.solve(M,b)                                                                  # Solving the matrix equation
And_polar = [polar(x) for x in Ans_rect]                                                        # Changing from rectangular form to polar form

for x in And_polar:
    x_list = list(x)
    x_list[1] = x_list[1]*180.0/pi
    x_tuple = tuple(x_list)
    And_polar[And_polar.index(x)] = x_tuple

# Printing the Voltages of nodes and Currents through the Voltage Sources
for i in range(1,N_size):
    print(" The Voltage at node {} has magnitude: {:.2e} V and Phase: {:.4f}".format(nodes[str(i)], And_polar[i-1][0], And_polar[i-1][1]))
for i in range(V_size):
    print(" The current through the Voltage source {} is {:.2e} A and Phase: {:.4f}".format(voltages['V'+str(i)], And_polar[i+N_size-1][0], And_polar[i+N_size-1][1]))
print('\nNote:')
print(' The Phase printed is in Degrees.')
print(' The Direction of current through Voltage source is from +ve terminal to -ve terminal')

    
########################### End Of Code #############################

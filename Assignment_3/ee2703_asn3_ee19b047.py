######################################### EE2703 Applied Programming Lab - Jan-May 2021########################################
###########################
######### Name          : Pnn Sumedh
######### Roll.No       : EE19B047
######### Assignment.No : 3
######### Commandline   : python ee2703_asn3_ee19b047.py <path to fitting.dat file>
###########################


# Importing required libraries
import numpy as np
import sys
import os.path                                                          
from os import path
from io import StringIO
import matplotlib.pyplot as plt
import scipy.special as sp
from scipy.linalg import lstsq

filepath = sys.argv[1]                                                                                          # Accessing file path from command argument            

if path.exists(filepath) == False:                                                                              # Checking for file in the given path
    print("Error! File with the given file name doesnot exist in the given path.")
    exit()                                                                                                      # Exiting the program if the file is not present

f = open(filepath)                                                                                              # Opening the file using file handling
fitting = f.read()                                                                                              # Reading the file line by line
f.close()

values = np.loadtxt(StringIO(fitting))                                                                          # Getting the values in Numpy array

time = values[:,0]                                                                                              # Storing time data points
sigma = np.logspace(-1,-3,9)                                                                                    # Storing Sigma of each function
func_dict = {}                                                                                                  # Creating a function dictionary
plt.figure(num=0)                                                                                               # Defining the figure to be zero
for i in range(1,10):
    func_dict['func'+str(i)] = values[:,i]
    plt.plot(time, func_dict['func'+str(i)], label = r'$\sigma_{}: {:.2e}$'.format(i,sigma[i-1]))               # Plotting each function on the figure 0                      


def g(t, A, B):                                                                                                 # Defining the function to find the function for given A and B
    y = A*sp.jn(2,t) + B*t
    return y
    
    
truevalue = g(time, 1.05, -0.105)                                                                               # Calculating the true function without noise

# Plotting the functions with noise and also the true function
plt.plot(time, truevalue, label = 'True Value',color = 'black')
plt.legend(fontsize = 5)
plt.xlabel('$t \longrightarrow$')
plt.ylabel('$f(t) + noise$')
plt.title('Q4: Data to be fitted to theory')

plt.show()

# Plotting the errorbar

plt.figure(num=1)
plt.errorbar(time[::5],func_dict['func1'][::5],sigma[0],fmt = 'ro', label = '$ErrorBar$')
plt.plot(time, truevalue, label = '$f(t)$')
plt.xlabel("$t \longrightarrow$")
plt.title("Q5: Data points for $\sigma = 0.10$ along with exact function")
plt.legend(fontsize = 10)
plt.show()

# Calculating M matrix

M = np.c_[sp.jn(2,time),time]

truevalue_matrix = np.dot(M,np.array([[1.05],[-0.105]])).reshape(-1)

# Checking whether both the vectors obtained are equal or not
if np.allclose(truevalue, truevalue_matrix):
    print(" The vectors obtained from equation and Matrix multiplication are same")
else:
    print(" The vectors obtained from equation and Matrix multiplication are different")

    
A_values = np.arange(0,2.1,0.1)                                                                                 # Defining the values of A
B_values = np.arange(-0.2,0.01,0.01)                                                                            # Defining the values of B
e = np.zeros((len(A_values),len(B_values)))
for i in range(A_values.shape[0]):
    for j in range(B_values.shape[0]):
        e[i,j] = np.sum(np.square(values[:,1] - g(time, A_values[i], B_values[j])))/101                         # Calculating error matrix

# Plotting Contour plots
index = np.unravel_index(np.argmin(e), e.shape)                                                                 # Finding the index where minimum mean squared error occurs         
fig, ax = plt.subplots()
CS = ax.contour(A_values,B_values,e,levels = 16)
ax.clabel(CS, CS.levels[:5], inline = 1, fontsize=10)
ax.plot(A_values[index[0]], B_values[index[1]], marker = 'o', color = 'r')
ax.annotate("Location of minimum", xy=(A_values[index[0]], B_values[index[1]]))
plt.xlabel(r'$A \longrightarrow$')
plt.ylabel(r'$B \longrightarrow$')
plt.title(r'Q8: Contour plot of $\epsilon_{ij}$')

plt.show()

AB_best_est = np.zeros((9,2))
AB = np.array([1.05,-0.105])

for i in range(1,10):
    best, *_ = lstsq(M, values[:,i])
    AB_best_est[i-1][0] = best[0]
    AB_best_est[i-1][1] = best[1]
# Plotting error in A and B
plt.figure(num=3)
AB_error = np.abs(AB_best_est - AB)
plt.plot(sigma,AB_error[:,0],color = 'r', linestyle = ':',markerfacecolor = 'r',marker = 'o',label = '$Aerr$')
plt.plot(sigma,AB_error[:,1],color = 'g', linestyle = ':',markerfacecolor = 'g',marker = 'o',label = '$Berr$')
plt.title('Q10: Variation of error with Noise')
plt.xlabel('$Noise$ $standard$ $deviation$ $\longrightarrow$')
plt.ylabel('$MS$ $error$ $\longrightarrow$')
plt.legend()
plt.show()

# Plotting error in A and B using loglog scale
plt.figure(num=4)
plt.stem(sigma,AB_error[:,0], use_line_collection=True,linefmt = 'r')
plt.stem(sigma,AB_error[:,1], use_line_collection=True,linefmt = 'g')
plt.loglog(sigma,AB_error[:,0],linestyle = 'None',marker = 'o',label = '$Aerr$',markerfacecolor = 'r',mew = 0)
plt.loglog(sigma,AB_error[:,1],linestyle = 'None',marker = 'o',label = '$Berr$',markerfacecolor = 'g',mew = 0)
plt.xlabel('$\sigma_n \longrightarrow$')
plt.ylabel('$MSerror$')
plt.legend()
plt.title('Q11: Variation of error with noise')
plt.show()


######################## End Of Code ###########################
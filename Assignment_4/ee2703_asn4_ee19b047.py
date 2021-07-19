######################################### EE2703 Applied Programming Lab - Jan-May 2021########################################
###########################
######### Name          : Pnn Sumedh
######### Roll.No       : EE19B047
######### Assignment.No : 4
######### Commandline   : python ee2703_asn4_ee19b047.py
###########################

# Importing Libraries

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.linalg import lstsq

########### Question 1

def exp(In):                                                                                                                # Defining function exp()
    flr = np.floor(In/(2*np.pi))
    In = In - (2*np.pi)*flr
    out = np.exp(In)
    return out


def coscos(In):                                                                                                             # Defining function cos(cos(x))
    temp = np.cos(In)
    out = np.cos(temp)
    return out

Input = np.linspace(-2*np.pi , 4*np.pi, 500)                                                                                # Creating numpy array with values between -2pi to 4pi divided into 500 points
plt.figure(num=1)                                                                                                           # Numbering the figure
plt.semilogy(Input, np.exp(Input), label = 'Actual Plot')                                                                   # Plotting actual graph of function exp() 
plt.semilogy(Input, exp(Input), label = 'Expected Plot')                                                                    # Plotting the expected graph of function exp()
plt.xlabel('Input $\longrightarrow$')
plt.ylabel('$e^{x} \longrightarrow$')
plt.grid()
plt.title(' Semilog plot of $e^x$ and $2\pi$ periodic $e^x$ ')
plt.legend()
plt.show()

plt.figure(num=2)                                                                                                           # Numbering the figure
plt.plot(Input, np.cos(np.cos(Input)), label = 'Actual Plot',color = 'r')                                                   # Plotting the actual graph of function cos(cos(x))
plt.plot(Input, coscos(Input), label = 'Expected Plot', linestyle = 'dashed',linewidth = '1.25')                            # Plotting the expected graph of fucntion cos(cos(x))
plt.xlabel('Input $\longrightarrow$')
plt.ylabel('$\cos (\cos (x))$')
plt.grid()
plt.title(' Plot of $\cos(\cos(x))$')
plt.legend()
plt.show()

##### Question 2

def coeff(func):                                                                                                            # Defining a function which takes a function as input and gives out Fourier Coefficients as output
    def uv(func):                                                                                                           # Defining a function to give (1/pi)*f(x)*cos(x) and (1/pi)*f(x)*sin(x) as output
        u = lambda x,k: func(x)*np.cos(k*x)/np.pi
        v = lambda x,k: func(x)*np.sin(k*x)/np.pi
        return u, v

    u, v = uv(func)                                                                                                         # Storing f(x)*cos(x) in u and f(x)*sin(x) in v
    a = [integrate.quad(u,0,2*np.pi,args=(k))[0] for k in range(26)]                                                        # Calculating the a coefficients and storing them in list a
    b = [integrate.quad(v,0,2*np.pi,args=(k))[0] for k in range(26)]                                                        # Calculating the b coefficients and storing them in list b
    a[0] = a[0]/2
    out = np.delete((np.array([[i,j] for i, j in zip(a,b)]).ravel()),1)                                                     # Storing a coefficients and b coefficients alternatively in out array
    return out

expo = lambda x: np.exp(x)                                                                                                  # Defining function expo
coscos = lambda x: np.cos(np.cos(x))                                                                                        # Defining function coscos
exp_vector = coeff(expo)                                                                                                    # Storing Fourier Coefficients of exp(x) in exp_vector
coscos_vector = coeff(coscos)                                                                                               # Storing Fourier Coefficients of cos(cos(x)) in coscos_vector

######## Question 4

x = np.linspace(0, 2*np.pi,401)                                                                                             # Creating an array with values between 0 and 2pi divided into 401 points
x = x[:-1]                                                                                                                  # Removing 2pi from the array of numbers
b_lstsq_exp = expo(x)                                                                                                       # Storing the value of exp(x) for the input values x
b_lstsq_coscos = coscos(x)                                                                                                  # Storing the value of cos(cos(x)) for the input values x
A = np.zeros((400,51))                                                                                                      # Creating a zero matrix of size 400x51
A[:,0] = 1                                                                                                                  # Storing 1 in first column of the matrix A
for k in range(1,26):
    A[:,2*k-1] = np.cos(k*x)                                                                                                # Storing the value in odd columns of matrix A
    A[:,2*k] = np.sin(k*x)                                                                                                  # Storing the value in even columns of matrix A

c_exp_lstsq = lstsq(A,b_lstsq_exp)[0]                                                                                       # Finding coefficients of exp(x) using least squares
c_coscos_lstsq = lstsq(A,b_lstsq_coscos)[0]                                                                                 # Finding coefficients of cos(cos(x)) using least squares



####### Question 3 and Question 5

plt.figure(num=3)                                                                                                           #Plotting Fourier Coefficients obtained for exp(x) from Both integration method and Least Squares method on semilog graph
n = np.array(list(range(51)))
plt.semilogy(n, np.abs(exp_vector), marker = 'o', linestyle = 'None',markerfacecolor = 'r',markeredgecolor = 'r',label = 'Integrated')
plt.semilogy(n, np.abs(c_exp_lstsq), marker = 'o', linestyle = 'None',markerfacecolor = 'g',markeredgecolor = 'g',label = 'Least Squares', markersize = 4)
plt.xlabel('n   $\longrightarrow$')
plt.ylabel('Coefficient  $\longrightarrow$')
plt.title(' Semilog plot of fourier coefficients of $e^x$')
plt.legend()
plt.grid()
plt.show()

plt.figure(num=4)                                                                                                           #Plotting Fourier Coefficients obtained for exp(x) from Both integration method and Least Squares method on loglog graph
plt.loglog(n, np.abs(exp_vector), marker = 'o', linestyle = 'None',markerfacecolor = 'r',markeredgecolor = 'r',label = 'Integrated')
plt.loglog(n, np.abs(c_exp_lstsq), marker = 'o', linestyle = 'None',markerfacecolor = 'g',markeredgecolor = 'g',label = 'Least Squares', markersize = 4)
plt.xlabel('n   $\longrightarrow$')
plt.ylabel('Coefficient  $\longrightarrow$')
plt.title(' log-log plot of fourier coefficients of $e^x$')
plt.legend()
plt.grid()
plt.show()

plt.figure(num=5)                                                                                                           #Plotting Fourier Coefficients obtained for cos(cos(x)) from Both integration method and Least Squares method on semilog graph
plt.semilogy(n, np.abs(coscos_vector), marker = 'o', linestyle = 'None',markerfacecolor = 'r',markeredgecolor = 'r',label = 'Integrated')
plt.semilogy(n, np.abs(c_coscos_lstsq), marker = 'o', linestyle = 'None',markerfacecolor = 'g',markeredgecolor = 'g',label = 'Least Squares', markersize = 4)
plt.xlabel('n   $\longrightarrow$')
plt.ylabel('Coefficient  $\longrightarrow$')
plt.title(' Semilog plot of fourier coefficients of $\cos(\cos(x))$')
plt.legend()
plt.grid()
plt.show()

plt.figure(num=6)                                                                                                           #Plotting Fourier Coefficients obtained for cos(cos(x)) from Both integration method and Least Squares method on loglog graph
plt.loglog(n, np.abs(coscos_vector), marker = 'o', linestyle = 'None',markerfacecolor = 'r',markeredgecolor = 'r',label = 'Integrated')
plt.loglog(n, np.abs(c_coscos_lstsq), marker = 'o', linestyle = 'None',markerfacecolor = 'g',markeredgecolor = 'g',label = 'Least Squares', markersize = 4)
plt.xlabel('n   $\longrightarrow$')
plt.ylabel('Coefficient  $\longrightarrow$')
plt.title(' Log-Log plot of fourier coefficients of $\cos(\cos(x))$')
plt.legend()
plt.grid()
plt.show()

########## Question 6

exp_deviation = np.abs(exp_vector - c_exp_lstsq)                                                                            # Calculating error vector of coefficients of exp(x) obtained from integration and least squares
coscos_deviation = np.abs(coscos_vector - c_coscos_lstsq)                                                                   # Calculating error vector of coefficients of cos(cos(x)) obtained from integration and least squares
exp_large_dev = np.amax(exp_deviation)                                                                                      # Finding the largest deviation for exp(x)
coscos_large_dev = np.amax(coscos_deviation)                                                                                # Finding the largest deviation for cos(cos(x))
print(' Largest Error for the fucntion exp(x): {}'.format(exp_large_dev))                                                   # Printing the values
print(' Largest Error for the fucntion cos(cos(x)): {}'.format(coscos_large_dev))   

########## Question 7

exp_lstsq = A.dot(c_exp_lstsq)                                                                                              # Finding the function data points obtained from least squares for exp(x)
coscos_lstsq = A.dot(c_coscos_lstsq)                                                                                        # Finding the function data points obtained from least squares for cos(cos(x))

plt.figure(num=1)                                                                                                           # Plotting the estimated and actual graph of exp(x)
plt.semilogy(x, exp_lstsq, linestyle='None', marker = 'o', markerfacecolor = 'g', markeredgecolor = 'g', label = ' Using least squares')
plt.semilogy(x, exp(x), color = 'r', label = 'True')
plt.xlabel('x $\longrightarrow$')
plt.ylabel('$e^x \longrightarrow$')
plt.title(' Semilog plot of $e^x$ from the coefficients from lstsq method')
plt.legend(fontsize = 15)
plt.grid()
plt.show()

plt.figure(num=2)                                                                                                           # Plotting the estimated and actual graph of cos(cos(x))
plt.plot(x, coscos_lstsq, linestyle = 'None', marker = 'o', markerfacecolor = 'g', markeredgecolor = 'g', label = ' Using least squares')
plt.plot(x, coscos(x), label = 'True', color = 'yellow')
plt.xlabel('x $\longrightarrow$')
plt.ylabel('$\cos(\cos(x)) \longrightarrow$')
plt.title(' Plot of $\cos(\cos(x))$ from the coefficients from lstsq method')
plt.legend()
plt.grid()
plt.show()

################################## End of Code ##########################################
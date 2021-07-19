######################################### EE2703 Applied Programming Lab - Jan-May 2021########################################
###########################
######### Name          : Pnn Sumedh
######### Roll.No       : EE19B047
######### Assignment.No : 6
######### Commandline   : python ee2703_asn6_ee19b047.py --n n_value --M M_value --nk nk_value --u0 u0_value --p p_value --Msig Msig_value
###########################

# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import argparse


# Input Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--n', help='The spatial size of grid', type = int, default = 100)
parser.add_argument('--M', help = 'Number of electrons injected in one turn', type = int, default=5)
parser.add_argument('--nk', help = 'Number of turns to simulate', type = int, default = 500)
parser.add_argument('--u0', help = 'Threshold Velocity', type = float, default = 5.)
parser.add_argument('--p', help = 'Probability that ionization will occur', type = float, default = 0.25)
parser.add_argument('--Msig', help = 'Standard Diveation', type = float, default = 2)

# Storing Input Arguments in variables
args = parser.parse_args()
n = args.n
M = args.M
nk = args.nk
u0 = args.u0
p = args.p
Msig = args.Msig

# Sanity check for probability being in [0,1]
if p<0 or p>1:
    print('Error! Probability should lie in [0,1]')
    exit()

# Sanity check for n, M, nk being positive
if n<0 or M<0 or nk<0:
    print('Error! n, M, nk all must be positive integers. Please enter the correct values.')
    exit()
# Defining the data vectors
xx = np.zeros(n*M)
u = np.zeros(n*M)
dx = np.zeros(n*M)

I = []  # Defining Intensity Vector
X = []  # Defining Position Vector
V = []  # Defining Velocity Vector

# Iteration
for k in range(1,nk):
    
    # Checking for electrons in the chamber
    ii = np.where(xx>0)
    dx[ii] = u[ii]+0.5
    xx[ii] = xx[ii]+dx[ii]
    u[ii] = u[ii]+1

    # Updating the values where electrons exited the chamber
    xx[np.where(xx>=n)] = 0
    u[np.where(xx>=n)] = 0
    dx[np.where(xx>=n)] = 0
    
    # Finding the electrons which are energitic and will collide
    kk = np.where(u>=u0)[0]
    ll = np.where(np.random.rand(len(kk))<=p)
    kl = kk[ll]
    
    # Updating the values where electrons have collided and there has been an emission
    u[kl] = 0
    xx[kl] = xx[kl] - dx[kl]*np.random.rand(len(kl))
    
    # Updating the intensity vector
    I.extend(xx[kl].tolist())
    
    # Injecting new electrons
    m = int(np.random.randn()*Msig+M)   # No of electrons entering in this iteration
    emptypos = np.where(xx==0)        # finding the empty positions in xx vector
    minindx = min(len(emptypos[0]),m) # Finding the minimum of m and len(mm)[0]
    xx[emptypos[0][:minindx]] = 1     # Updating the position for the new electrons
    u[emptypos[0][:minindx]] = 0      # Updating the velocity of new electrons
    
    X.extend(xx[np.where(xx>0)].tolist())   # Updating the position vector 
    V.extend(u[np.where(xx>0)].tolist())    # Updating the velocity vector
    
# Plotting Electron Phase Space    
plt.figure(2)
plt.plot(X,V,'bo',markersize = 1)
plt.title('Electron Phase Space')
plt.xlabel('X $\longrightarrow$')
plt.ylabel('V $\longrightarrow$')

# Plotting Intensity Histogram
plt.figure(1)
count, bins, _ = plt.hist(I,n, [0,n], histtype='bar', edgecolor='black')
plt.title('Light Intensity')

# Plotting Electron Density
plt.figure(0)
plt.hist(X, n, [0,n] , histtype='bar', edgecolor='black')
plt.title('Electron Density')
plt.show()
plt.show()
plt.show()

# Printing Intensity Table
xpos = 0.5*(bins[0:-1]+bins[1:])
intensitycount = [[x,y] for x, y in zip(xpos, count)]
print("Intensity Data : \n")
print(tabulate((intensitycount), headers=['xpos', 'count'],tablefmt='github'))

############################### End of Code ##################################
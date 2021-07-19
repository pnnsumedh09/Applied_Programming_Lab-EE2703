######################################### EE2703 Applied Programming Lab - Jan-May 2021########################################
###########################
######### Name          : Pnn Sumedh
######### Roll.No       : EE19B047
######### Assignment.No : 5
######### Commandline   : python ee2703_asn5_ee19b047.py Nx Ny radius iter
###########################

# Importing Libraries
from pylab import *
import mpl_toolkits.mplot3d.axes3d as p3
from scipy.linalg import lstsq
from mpl_toolkits import mplot3d
import os,sys

# Input Arguments
if(len(sys.argv)==5):
    Nx=int(sys.argv[1])
    Ny=int(sys.argv[2])
    radius=float(sys.argv[3])  
    Niter=int(sys.argv[4])
    if radius>0.5:
        print("\nError!!! Radius out of range. Give a value between 0 and 0.5\n")
        exit()
    
else:
    Nx=25 # size along x
    Ny=25 # size along y
    radius=0.35 #radius of central lead
    Niter=1500 #number of iterations to perform

phi = np.zeros((Ny,Nx)) # Initializing phi array

# Plotting the contour plot of potential

x = np.linspace(-0.5, 0.5, num = Nx)    
y = np.linspace(-0.5, 0.5, num = Ny)[::-1]

X,Y = meshgrid(x,y) # Defining Meshgrid

phi[np.where(X*X + Y*Y <= radius**2)] = 1.0 # Initializing the center part of phi array = 1V
X_con = X[np.where(X*X + Y*Y <= radius**2)] # Finding the X-location of elements which are inside the wire
Y_con = Y[np.where(X*X + Y*Y <= radius**2)] # Finding the Y-location of elements which are inside the wire

plt.figure(1) 
plt.contourf(X,Y, phi)    # Plotting the contour plot of initial potential
plt.plot(X_con, Y_con , 'o', color = 'red')  # Plotting red dots in the region enclosed by wire
plt.title('Contour Plot of Potential')
plt.xlabel('X')
plt.ylabel('Y')
plt.colorbar()
plt.show()

# Performing the Iteration

errors = np.zeros(Niter)    # Initializing error array
for k in range(Niter):
    oldphi = phi.copy()
    phi[1:-1,1:-1] = 0.25*(phi[1:-1, 0:-2]+phi[1:-1, 2:]+phi[0:-2, 1:-1]+phi[2:, 1:-1]) # Taking average of neighbour elements
    phi[1:-1,0] = phi[1:-1,1]   # Applying Boundary conditions
    phi[1:-1, -1] = phi[1:-1, -2]   # Applying Boundary Conditions
    phi[0, 1:-1] = phi[1, 1:-1] # Applying Boundary Conditions
    phi[np.where(X*X + Y*Y <= radius**2)] = 1.0 # Reassigning 1V to center elements
    errors[k] = np.max(np.abs(phi-oldphi)) # Calculating error

plt.figure(num=2)
plt.semilogy(range(Niter), errors )
plt.xlabel('Iterations')
plt.ylabel('Error')
plt.title('Error On SemiLog Plot')
plt.show()
plt.figure(num=3)
plt.loglog(range(Niter), errors, label= 'Error' )
plt.loglog(range(Niter)[::50], errors[::50],'o', label = 'Every 50th Datapoint')
plt.xlabel('Interations')
plt.ylabel('Error')
plt.title('Error on LogLog Plot')
plt.legend()
plt.show()

# Plotting Error Plots

def fitting(err, ite):              # Function to find the Best Fit for the error function
    ite = ite.reshape(len(ite),1)
    ones = np.ones(np.shape(ite))
    M = np.hstack((ones, ite))
    b = np.log(err)
    out = lstsq(M, b)[0]
    B = out[1]
    A = np.exp(out[0])
    return A, B

iteration = np.array(list(range(Niter)))
fit1_vals = fitting(errors, iteration)
fit2_vals = fitting(errors[500:], iteration[500:])
fit1 = fit1_vals[0]*np.exp(fit1_vals[1]*iteration)
fit2 = fit2_vals[0]*np.exp(fit2_vals[1]*iteration)
plt.figure(num=4)
plt.loglog(iteration, errors, label = 'errors')
plt.loglog(iteration[::50], fit1[::50], 'o', color = 'r', label = 'fit1')
plt.loglog(iteration[::50], fit2[::50], 'o', color = 'g', label = 'fit2')
plt.xlabel('Iterations')
plt.ylabel('Error')
plt.title('Best Fit for Error (LogLog Plot)')
plt.legend()
plt.show()

# Surface Plot of Potential

fig = plt.figure(5)
ax = plt.axes(projection = '3d')
surf = ax.plot_surface(X,Y,phi, cmap = cm.jet, rstride=1, cstride=1)
ax.set_title('The 3D surface plot of the Potential')
plt.show()

# Contour Plot of Potential

plt.figure(num=6)
plt.contourf(X,Y, phi)
plt.plot(X_con, Y_con, 'o', color = 'red')
plt.title('Contour Plot of the Potential')
plt.colorbar()
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

#  Vector Plot of Currents
Jx = np.zeros((Ny,Nx))
Jy = Jx.copy()
Jx[1:-1,1:-1] = 0.5*(phi[1:-1,0:-2] - phi[1:-1, 2:])
Jy[1:-1,1:-1] = 0.5*(phi[2:,1:-1] - phi[0:-2, 1:-1])
plt.figure(7)
plt.plot(X_con, Y_con, 'o', color = 'red')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Vector Plot of Currents')
plt.quiver(X,Y,Jx,Jy, scale=5)
plt.show()

########################## End of Code ###############################
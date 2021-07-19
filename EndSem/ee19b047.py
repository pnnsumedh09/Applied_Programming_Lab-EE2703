########### Name: P N Neelesh Sumedh
########### Roll.No: EE19B047
########### End Semester Examination

# Importing the required libraries
import numpy as np
import matplotlib.pyplot as plt
PI = np.pi


# ### Pseudo Code:
# 
# SET x,y and z values 
# SET meshgrid using x,y,z 
# SET radius := 10, no_of_sections := 100 
# DETERMINE angular positions of each section of wire
# DETERMINE x,y coordinates of each section of wire 
# COMPUTE current flowing through each section and find x component and y component of direction of current flow
# PLOT quiver plot of the current flow
# DETERMINE length of each section
# DETERMINE x,y and z components of dl vector which points in the direction of current flow
# FUNCTION calc
#         PASS IN: index of a section of wire
#         DETERMINE an array of size (x by y by z by 3) which stores all the points in the space
#         COMPUTE an array of size (x by y by z) which stores the distance between the section of wire and points of space
#         COMPUTE the vector potential due to this section of wire at all points of space
#         PASS OUT: Vector potential array
# ENDFUNCTION
# 
# INITIALIZE A:= a zero array
# FOR every index of section of wire
#         CALL: calc and find Vector Potential in space due to this section of wire
#         UPDATE A by adding the Vector Potential in space due to this section to A
# ENDFOR
# COMPUTE x component of A in every point on space
# COMPUTE y component of A in every point on space
# COMPUTE B using the x and y components of A in space
# PLOT loglog plot of B varying with z
# ESTIMATE b and c using least squares method assuming B = c*z^b
# PRINT b and c
# PLOT the estimated magnetic field along with Original Magnetic Field


# Question 2
'''
Defining x,y,z coordinates.
Since we require only x and y = -1,0,1 and for z we require z = 1,2,.....,1000
we use linspace to get these values.
Then we create a meshgrid of X,Y,Z for these set of coordinate values.
'''
x = np.linspace(0,2,num=3)
y = np.linspace(0,2,num=3)
z = np.linspace(1,1000,num=1000)

X,Y,Z = np.meshgrid(x,y,z)


# Question 3

'''
1) Storing radius, no of sections, and angle of each sections in variables.
2) The angular position of each section is stored in the variable "angles".
3) The cosines of these angular positions is stored in "cosine" and sines of these angular positions is stored in "sine"
4) The x and y coordinates of each section is stored in an array "wire_coords" with first colomn storing x coordinates and second coloumn storing y coordinates.
5) The vector current is stored in the array "I". The x direction of this current is stored in first column and y direction of this current is stored in second column.
6) Using the "wire_coords" and "I" arrays, the current flow plot is plotted using 'quiver'.
'''
radius = 10
no_of_sections = 100
angle_section = 2*PI/no_of_sections
angles = np.linspace(0,2*PI,no_of_sections+1)[:-1]
cosine = np.cos(angles)
sine = np.sin(angles)
wire_coords = np.array([radius*cosine,radius*sine])
I = np.array([-1e7*sine*cosine,1e7*cosine*cosine])


plt.figure(num = 1,figsize=(8,8))
plt.plot(0,0,'ro',markersize=3)
plt.plot(wire_coords[0],wire_coords[1],'go',label='Centers of sections')
plt.xlabel('X$\longrightarrow$')
plt.ylabel('Y$\longrightarrow$')
plt.annotate('Center',[0,0],xytext=(-0.5,-0.5))
plt.legend()
plt.title('Centers of Sections of wire')
plt.grid()
plt.savefig('fig1.png')


plt.figure(num=2,figsize=(10,8))
colors = np.sqrt(I[0]**2 + I[1]**2)
Q = plt.quiver(wire_coords[0],wire_coords[1],I[0],I[1],colors,width=0.005,cmap='plasma',scale=1e8/1.5)
plt.plot(0,0,'ro',markersize=3)
plt.annotate('Center',[0,0],xytext=(-0.5,-0.5))
plt.text(-7.5,-5,'2) The direction of arrows shows the direction of current flow')
plt.text(-9,-2.5,'1) The magnitude of current is given by the color referenced to colorbar')
plt.quiver(wire_coords[0],wire_coords[1],I[0],I[1],edgecolor='black',facecolor = 'None',linewidth=0.3,width=0.005,scale=1e8/1.5)
plt.xlabel('X$\longrightarrow$')
plt.ylabel('Y$\longrightarrow$')
plt.colorbar(Q)
plt.title('Current Flow in the closed loop')
plt.grid()
plt.savefig('fig2.png')



# Question 4
'''
1) The length of each small section of loop is stored in "dl_length"
2) The x,y and z coordinates of the center of these small sections is stored in array "r_". The z coordinate is zero.
3) The x,y and z directions to which these small sections point to is stored in array "dl". The magnitude on z direction is zero.
'''
dl_length = 2*PI*radius/no_of_sections
r_ = np.array([radius*cosine,radius*sine])
r_ = np.append(r_,np.zeros((1,len(r_[0]))),axis=0).T
dl = np.array([-dl_length*sine,dl_length*cosine])
dl = np.append(dl,np.zeros((1,len(dl[0]))),axis=0).T


# Question 5 and 6
'''
1) Function calc() takes the index of the small segment as input.
2) The output of this function is the vector potential at all the points in space due to this small segment. Output is an array 'A' of size 3x3x3x1001 
3) The value of A in a particular direction is given by axis 0. A[0] gives in x-direction, A[1] in y and A[2] in z-direction.
4) The value of A at a point (x,y,z) is given by axis 1,2 and 3. A[:,x,y,z] gives A at point (x,y,z) in all the three directions.
5) The array "r" has a shape 3x3x1001x3. This stores all the points in space. To get a point (x,y,z), we write r[x,y,z].
6) The array "R" has a shape 3x3x1001. This stores the distance between the segment and all the points in space. Distance between the segment and (x,y,z) is given by R[x,y,z]
'''
def calc(l):       
    r = np.moveaxis(np.array((X,Y,Z)).T,[0,1,2],[2,0,1])
    R = np.linalg.norm(r - r_[l],axis=3)
    A = np.exp(-1j*R/radius)/R*cosine[l]*(dl[l].reshape(3,1,1,1))
    return A      # [A in which axis][Point at which A is required (x,y,z)]

# Question 7
'''
1) The Vector Potential due to all sections in the space is added together to get the Resultant Vector Potential
2) The x components of Vector potential are stores in A_x
3) The y components of Vector potential are stored in A_y
4) For the x component of Vector potential at point (x,y,z), we write A_x[x,y,z]. Similarly for y-component.
'''
A = np.zeros(calc(0).shape)
for l in range(no_of_sections):
    A = calc(l)+A
A_x = A[0]
A_y = A[1]

# Question 8 and 9

'''
1) The Magnetic field is computed by the Numerical Method from the Vector Potentials
2) The Computed Magnetic Field is plotted on loglog scale

'''
B=(A_y[2,1,:]-A_x[1,2,:]-A_y[0,1,:]+A_x[1,0,:])/(2.0)
plt.figure(num = 3,figsize=(8,8))
plt.loglog(z,np.abs(B),label='Magnetic Field Variation with z')
plt.title('Variation of Magnetic Field with z on loglog plot')
plt.legend()
plt.xlabel('z $\longrightarrow$')
plt.ylabel('Magnetic Field $\longrightarrow$')
plt.grid()
plt.savefig('fig3.png')




# Question 10
'''
1) The Magnetic field is approximated to be an exponential in z: B = c*z^b
2) The log-log format of this is: log(B) = log(c) + b*log(z)
3) The matrix S is used to store the values of log(z) along with a column with 1 s.
4) lstsq is used to estimate values of log(c) and b
5) c is calculated from log(c)
'''
S = np.hstack([np.ones((len(B[20:]),1)),(np.log(z[20:])).reshape(len(B[20:]),1)])
log_c, b = np.linalg.lstsq(S,np.log(np.abs(B[20:])),rcond=None)[0]
c = np.exp(log_c)
print("The Estimated Value of b: {}".format(b))
print("The Estimated Value of c: {}".format(c))


'''
1) The B_est is the estimated Magnetic Field: c*z^b
2) Plotting the Original Magnetic Field and Estimated Magnetic Field
'''
B_est = c*z**b
plt.figure(num=4,figsize=(8,8))
plt.loglog(z,np.abs(B),label='Magnetic Field computed from vector potential')
plt.loglog(z,B_est,label='Estimated Magnetic Field')
plt.legend()
plt.grid()
plt.title("Loglog plot of Original Magnetic field and Estimated Magnetic Field")
plt.xlabel('z$\longrightarrow$')
plt.ylabel('Magnetic Field$\longrightarrow$')
plt.savefig('fig4.png')
plt.show()

############# END OF CODE
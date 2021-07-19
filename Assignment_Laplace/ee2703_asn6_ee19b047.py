######################################### EE2703 Applied Programming Lab - Jan-May 2021########################################
###########################
######### Name          : Pnn Sumedh
######### Roll.No       : EE19B047
######### Assignment.No : 6
######### Commandline   : python ee2703_asn6_ee19b047.py 
###########################

import scipy.signal as sp
import numpy as np
import scipy
import matplotlib.pyplot as plt

def springtransfun(freq,damping):
    return sp.lti([1,damping],np.polymul([1.0,0,freq**2],[1,2*damping,freq**2+damping**2]))
# Question 6
L = 1e-6
C = 1e-6
R = 100
H = sp.lti([1],[L*C,R*C,1])
time = np.linspace(0,3e-2,10000)
v = np.cos(1e3*time) - np.cos(1e6*time)
t,y,svec = sp.lsim(H,v,time)
plt.figure(10)
plt.title('Output of RLC circuit for t<30 msec')
plt.xlabel('t $\longrightarrow$')
plt.ylabel('$V_{out}$ $\longrightarrow$')
plt.plot(t,y)

time = np.linspace(0,30e-6,10000)
v = np.cos(1e3*time) - np.cos(1e6*time)
t,y,svec = sp.lsim(H,v,time)
plt.figure(9)
plt.title('Output of RLC circuit for t<30 usec')
plt.xlabel('t $\longrightarrow$')
plt.ylabel('$V_{out}$ $\longrightarrow$')
plt.plot(t,y)

# Question 5
w, S, phi = H.bode()
figure, (ax1,ax2) = plt.subplots(2,1,num=8)
ax1.semilogx(w,S)
ax1.set_title('Magnitude')
ax2.semilogx(w,phi)
ax2.set_title('Phase')
figure.tight_layout()

# Question 4    
X = sp.lti([1,0,2],[1,0,3,0])
Y = sp.lti([2],[1,0,3,0])
t,x = sp.impulse(X,None,np.linspace(0,20,5001))
t,y = sp.impulse(Y,None,np.linspace(0,20,5001))
plt.figure(7)
plt.plot(t,x,label = 'x')
plt.title('Coupled Oscillations X and Y')
plt.xlabel('t $\longrightarrow$')
plt.ylabel('Amplitude $\longrightarrow$')
plt.plot(t,y, label = 'y')
plt.legend()

# Question 3
X_F = sp.lti([1],[1,0,2.25])
i = 6
for f in np.arange(1.6,1.35,-0.05):
    t = np.linspace(0,150,5001)
    inp = np.cos(f*t)*np.exp(-0.05*t)
    t,y,svec = sp.lsim(X_F,inp,t)
    plt.figure(i)
    plt.plot(t,y)
    plt.title('Forced Damping Oscillator with frequency = '+str(f))
    plt.xlabel('t $\longrightarrow$')
    plt.ylabel('x $\longrightarrow$')
    i -= 1

# Question 2
X = springtransfun(1.5,0.05)
t,x = sp.impulse(X,None,np.linspace(0,50,5001))
plt.figure(1)
plt.title('Forced Damping Oscillator with damping = 0.05')
plt.xlabel('t $\longrightarrow$')
plt.ylabel('x $\longrightarrow$')
plt.plot(t,x)

    
# Question 1
X = springtransfun(1.5,0.5)
t,x = sp.impulse(X,None,np.linspace(0,50,5001))
plt.figure(0)
plt.title('Forced Damping oscillator with damping = 0.5')
plt.xlabel('t $\longrightarrow$')
plt.ylabel('x $\longrightarrow$')
plt.plot(t,x)

for i in range(11):
    plt.show()
    
######################## End of Code ######################
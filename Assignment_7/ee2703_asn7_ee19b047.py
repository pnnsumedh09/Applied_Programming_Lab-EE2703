import sympy as sy
import pylab as p
import scipy.signal as sp
import numpy as np
import warnings

warnings.filterwarnings("ignore")
sy.init_session

def lowpassfilter(R1,R2,C1,C2,G,Vi):
    s = sy.symbols('s')
    A = sy.Matrix([[0,0,1,-1/G],[-1/(1+s*R2*C2),1,0,0],[0,-G,G,1],[-1/R1-1/R2-s*C1,1/R2,0,s*C1]])
    b = sy.Matrix([0,0,0,-Vi/R1])
    V = A.inv()*b
    return (A,b,V)

def highpassfilter(R1,R2,C1,C2,G,Vi):
    s = sy.symbols('s')
    A = sy.Matrix([[0,0,1,-1/G],[-s*R2*C2/(1+s*R2*C2),1,0,0],[0,-G,G,1],[-s*C1-s*C2-1/R1,s*C2,0,1/R1]])
    b = sy.Matrix([0,0,0,-Vi*s*C1])
    V = A.inv()*b
    return (A,b,V)

def TFconvt(h):
    s = sy.symbols('s')
    n, d = sy.fraction(h)
    N = sy.Poly(n, s).all_coeffs()
    D = sy.Poly(d, s).all_coeffs()
    N, D = [float(f) for f in N], [float(f) for f in D]
    return sp.lti(N, D)

# Question 1
s = sy.symbols('s')
(A1,b1,V1) = lowpassfilter(10000,10000,1e-9,1e-9,1.586,1/s)
H1 = TFconvt(V1[3])
t,SR1 = sp.impulse(H1,None,np.linspace(0,1e-3,10000))
p.plot(t,SR1)
p.grid(True)
p.xlabel(r'$t$')
p.ylabel(r'$v_o(t)$')
p.title('Step response of LPF')
p.show()

# Question 2
t2 = np.linspace(0,1e-3,1000000)
vi2 = np.sin(2000*np.pi*t2)+np.cos(2e6*np.pi*t2)
(A2,b2,V2) = lowpassfilter(10000,10000,1e-9,1e-9,1.586,1)

H2 = TFconvt(V2[3])
t2,Vo2,svec = sp.lsim(H2,vi2,t2)
p.plot(t2,vi2)
p.plot(t2,Vo2)
p.grid()
p.xlabel(r'$t$')
p.ylabel('V')
p.title('LPF output for sum of sinusoids along with input')
p.show()

# Question 3
s = sy.symbols('s')
(A3,b3,V3) = highpassfilter(10000,10000,1e-9,1e-9,1.586,1)

w = p.logspace(1,8,300)
ss = 1j*w
f = sy.lambdify(s,V3[3],'numpy')
p.loglog(w,abs(f(ss)))
p.grid()
p.xlabel(r'$\omega$')
p.ylabel(r'$|H(j\omega)|$')
p.title('Magnitude Response of HPF')
p.show()

# Question 4
H3 = TFconvt(V3[3])
t3 = np.linspace(0,1e-2,30000)
vi3 = np.cos(1.7e6*np.pi*t3)*np.exp(-100*t3)
t3,Vo3,svec = sp.lsim(H3,vi3,t3)
p.plot(t3,Vo3,label='output')
p.legend()
p.grid()
p.xlabel(r'$t$')
p.ylabel(r'$v_o(t)$')
p.title('Response of HPF to a high frequency damped sinusoid')
p.show()

t3 = np.linspace(0,1e-3,3000)
vi3 = np.cos(1e3*np.pi*t3)*np.exp(-100*t3)
t3,Vo3,svec = sp.lsim(H3,vi3,t3)
p.plot(t3,Vo3,label='output')
p.legend()
p.grid()
p.xlabel(r'$t$')
p.ylabel(r'$v_o(t)$')
p.title('Response of HPF to a low frequency damped sinusoid')
p.show()

# Question 5
(A5,b5,V5) = highpassfilter(10000,10000,1e-9,1e-9,1.586,1/s)
H5 = TFconvt(V5[3])
t,SR = sp.impulse(H5,None,np.linspace(0,1e-3,100000))
p.plot(t,SR)
p.xlabel(r'$t$')
p.ylabel(r'$v_o(t)$')
p.grid()
p.title('Step Response of HPF')
p.show()
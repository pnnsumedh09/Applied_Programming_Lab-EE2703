import pylab as p

# Question 1 Working out through Examples

x = p.linspace(0,2*p.pi,128)
y = p.sin(5*x)
Y = p.fft(y)
p.figure(figsize=(8,8))
p.subplot(2,1,1)
p.plot(abs(Y),lw=2)
p.ylabel(r"$|Y|$",size=16)
p.title(r"Spectrum of $\sin(5t)$")
p.grid()
p.subplot(2,1,2)
p.plot(p.unwrap(p.angle(Y)),lw=2)
p.grid()
p.ylabel(r"Phase of $Y$",size=16)
p.xlabel(r"$\omega$",size=16)
p.savefig('test1.png')
p.show()


x = p.linspace(0,2*p.pi,129);x=x[:-1]
y = p.sin(5*x)
w = p.linspace(-64,63,128)
Y = p.fftshift(p.fft(y))/128.0
p.figure(figsize=(8,8))
p.subplot(2,1,1)
p.plot(w,abs(Y),lw=2)
p.xlim([-10,10])
p.ylabel(r"$|Y|$",size=16)
p.title(r"Spectrum of $\sin(5t)$ corrected")
p.grid()

p.subplot(2,1,2)
p.plot(w,p.angle(Y),'o',lw=2,color='r')
ii = p.where(abs(Y)>1e-3)
p.plot(w[ii],p.angle(Y[ii]),'o',lw=2,color='g')
p.xlim([-10,10])
p.ylabel(r"Phase of $Y$",size=16)
p.xlabel(r"$\omega$",size=16)
p.grid(True)
p.savefig("test2.png")
p.show()


x = p.linspace(0,2*p.pi,129);x=x[:-1]
y = (1+0.1*p.cos(x))*p.cos(10*x)
w = p.linspace(-64,63,128)
Y = p.fftshift(p.fft(y))/128.0
p.figure(figsize=(8, 8))
p.subplot(2,1,1)
p.plot(w,abs(Y),lw=2)
p.xlim([-15,15])
p.ylabel(r"$|Y|$",size=16)
p.title(r"Spectrum of $\left(1+0.1\cos\left(t\right)\right)\cos\left(10t\right)$")
p.grid()

p.subplot(2,1,2)
ii = p.where(abs(Y)>1e-3)
p.plot(w[ii],p.angle(Y[ii]),'o',lw=2,color='r')
p.xlim([-15,15])
p.ylabel(r"Phase of $Y$",size=16)
p.xlabel(r"$\omega$",size=16)
p.ylim([-p.pi, p.pi])
p.grid(True)
p.savefig("test3.png")
p.show()


x = p.linspace(-4*p.pi,4*p.pi,513);x=x[:-1]
y = (1+0.1*p.cos(x))*p.cos(10*x)
w = p.linspace(-64,64,513); w = w[:-1]
Y = p.fftshift(p.fft(y))/512.0
p.figure(figsize=(8, 8))
p.subplot(2,1,1)
p.plot(w,abs(Y),lw=2)
p.xlim([-15,15])
p.ylabel(r"$|Y|$",size=16)
p.title(r"Spectrum of $\left(1+0.1\cos\left(t\right)\right)\cos\left(10t\right)$ corrected")
p.grid()

p.subplot(2,1,2)
ii = p.where(abs(Y)>1e-3)
p.plot(w[ii],p.angle(Y[ii]),'o',lw=2,color='r')
p.xlim([-15,15])
p.ylabel(r"Phase of $Y$",size=16)
p.xlabel(r"$\omega$",size=16)
p.ylim([-p.pi, p.pi])
p.grid(True)
p.savefig("test4.png")
p.show()


# Question 2a

x = p.linspace(-4*p.pi,4*p.pi,513);x=x[:-1]
y = p.sin(x)**3
w = p.linspace(-64,64,513); w = w[:-1]
Y = p.fftshift(p.fft(y))/512.0
p.figure(figsize=(8, 8))
p.subplot(2,1,1)
p.plot(w,abs(Y),lw=2)
p.xlim([-9,9])
p.ylabel(r"$|Y|$",size=16)
p.title(r"Spectrum of $\sin^3(t)$")
p.grid()

p.subplot(2,1,2)
ii = p.where(abs(Y)>1e-3)
p.plot(w[ii],p.angle(Y[ii]),'o',lw=2,color='g')
p.xlim([-9,9])
p.ylabel(r"Phase of $Y$",size=16)
p.xlabel(r"$\omega$",size=16)
p.ylim([-p.pi, p.pi])
p.grid(True)
p.savefig("Q2_a.png")
p.show()


# Question 2b

x = p.linspace(-4*p.pi,4*p.pi,513);x=x[:-1]
y = p.cos(x)**3
w = p.linspace(-64,64,513); w = w[:-1]
Y = p.fftshift(p.fft(y))/512.0
p.figure(figsize=(8, 8))
p.subplot(2,1,1)
p.plot(w,abs(Y),lw=2)
p.xlim([-9,9])
p.ylabel(r"$|Y|$",size=16)
p.title(r"Spectrum of $\cos^3(t)$")
p.grid()

p.subplot(2,1,2)
ii = p.where(abs(Y)>1e-3)
p.plot(w[ii],p.angle(Y[ii]),'o',lw=2,color='g')
p.xlim([-9,9])
p.ylabel(r"Phase of $Y$",size=16)
p.xlabel(r"$\omega$",size=16)
p.ylim([-p.pi, p.pi])
p.grid(True)
p.savefig("Q2_b.png")
p.show()

# Question 3

x = p.linspace(-4*p.pi,4*p.pi,513);x=x[:-1]
y = p.cos(20*x+5*p.cos(x))
w = p.linspace(-64,64,513); w = w[:-1]
Y = p.fftshift(p.fft(y))/512.0
p.figure(figsize=(8, 8))
p.subplot(2,1,1)
p.plot(w,abs(Y),lw=2)
p.xlim([-35,35])
p.ylabel(r"$|Y|$",size=16)
p.title(r"Spectrum of $\cos(20t + 5\cos(t))$")
p.grid()

p.subplot(2,1,2)
ii = p.where(abs(Y)>1e-3)
p.plot(w[ii],p.angle(Y[ii]),'o',lw=2,color='g')
p.xlim([-35,35])
p.ylabel(r"Phase of $Y$",size=16)
p.xlabel(r"$\omega$",size=16)
p.grid(True)
p.ylim([-4, 4])
p.savefig("Q3.png")
p.show()


# Question 4

T = 8*p.pi
samp_freq = 8/512
min_error = 1e-6
error = 1+min_error
Yold = 0
while error>min_error:
    N = int(T/p.pi/samp_freq)
    x = p.linspace(-T/2,T/2,N+1);x=x[:-1]
    y = p.exp(-0.5*x**2)
    w = p.linspace(-1/samp_freq,1/samp_freq,N+1); w = w[:-1]
    Y = p.fftshift(p.fft(p.ifftshift(y)))*T/N
    error = sum(abs(Y[::2]-Yold))
    Yold = Y
    T = 2*T
    print(error)
actual_dft = p.sqrt(2*p.pi)*p.exp(-0.5*w**2)
p.figure(figsize=(8, 8))

p.subplot(2,1,1)
p.plot(w,abs(Y),lw=3.5,label='Estimate')
p.plot(w,abs(actual_dft),'--',lw=3.5,label='Expected')
p.xlim([-5,5])
p.ylabel(r"$|Y|$",size=16)
p.legend()
p.title(r"Spectrum of $e^{(\frac{-t^2}{2})}$",size = 16)
p.grid()

p.subplot(2,1,2)
ii = p.where(abs(Y)>1e-3)
p.plot(w[ii],p.angle(Y[ii]),'o',markersize=7,color='g',label='Estimated')
ii = p.where(abs(actual_dft)>1e-3)
p.plot(w[ii],p.angle(actual_dft[ii]),'ro',markersize=4,label='Expected')
p.xlim([-5,5])
p.legend()
p.ylabel(r"Phase of $Y$",size=16)
p.xlabel(r"$k$",size=16)
p.grid(True)
p.ylim([-p.pi, p.pi])
p.savefig("fig4.png")
p.show()




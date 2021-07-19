import pylab as p
from pylab import pi as PI
import matplotlib.cm as cm
import mpl_toolkits.mplot3d.axes3d as p3

# Example 1

t = p.linspace(-p.pi,p.pi,65);t=t[:-1]
dt = t[1]-t[0];fmax=1/dt
y = p.sin(p.sqrt(2)*t)
y[0] = 0
y = p.fftshift(y)
Y = p.fftshift(p.fft(y))/64.0
w = p.linspace(-p.pi*fmax,p.pi*fmax,65);w=w[:-1]
p.figure(1,figsize = (8,8))
p.subplot(2,1,1)
p.plot(w,abs(Y),lw=2)
p.xlim([-10,10])
p.ylabel(r"$|Y|$",size=16)
p.title(r"Spectrum of $\sin\left(\sqrt{2}t\right)$")
p.grid(True)
p.subplot(2,1,2)
p.plot(w,p.angle(Y),'ro',lw=2)
p.xlim([-10,10])
p.ylabel(r"Phase of $Y$",size=16)
p.xlabel(r"$\omega$",size=16)
p.grid(True)
p.savefig("fig10-1.png")
p.show()

# Example 2

t1 = p.linspace(-PI,PI,65)[:-1]
t2 = p.linspace(-3*PI,-PI,65)[:-1]
t3 = p.linspace(PI,3*PI,65)[:-1]

p.figure(2,figsize=(8,8))
p.plot(t1,p.sin(p.sqrt(2)*t1),'b')
p.plot(t2,p.sin(p.sqrt(2)*t2),'r')
p.plot(t3,p.sin(p.sqrt(2)*t3),'r')
p.ylabel("$y$",size=16)
p.xlabel("$t$",size=16)
p.title(r"$\sin\left(\sqrt{2}t\right)$")
p.grid(True)
p.savefig("fig10-2.png")
p.show()

# Example 3

t1 = p.linspace(-PI,PI,65)[:-1]
t2 = p.linspace(-3*PI,-PI,65)[:-1]
t3 = p.linspace(PI,3*PI,65)[:-1]
y = p.sin(p.sqrt(2)*t1)
p.figure(3,figsize=(8,8))
p.plot(t1,y,'bo')
p.plot(t2,y,'ro')
p.plot(t3,y,'ro')
p.ylabel("$y$",size=16)
p.xlabel("$t$",size=16)
p.title(r"$\sin\left(\sqrt{2}t\right)$ with $t$ wrapping every $2\pi$ ")
p.grid(True)
p.savefig("fig10-3.png")
p.show()

# Example 4

t = p.linspace(-PI,PI,65);t=t[:-1]
dt = t[1]-t[0]; fmax = 1/dt
y = t
y[0] = 0 
y = p.fftshift(y)
Y = p.fftshift(p.fft(y))/64.0
w = p.linspace(-PI*fmax,PI*fmax,65);w = w[:-1]
p.figure(4,figsize=(8,8))
p.semilogx(abs(w),20*p.log10(abs(Y)),lw=2)
p.xlim([1,10])
p.ylim([-20,0])
p.xticks([1,2,5,10],["1","2","5","10"],size=16)
p.ylabel(r"$|Y|$ (dB)",size=16)
p.title(r"Spectrum of a digital ramp")
p.xlabel(r"$\omega$",size=16)
p.grid(True)
p.savefig("fig10-4.png")
p.show()

# Example 5

t1 = p.linspace(-PI,PI,65)[:-1]
t2 = p.linspace(-3*PI,-PI,65)[:-1]
t3 = p.linspace(PI,3*PI,65)[:-1]
n = p.arange(64)
wnd = p.fftshift(0.54+0.46*p.cos(2*PI*n/63))
y = p.sin(p.sqrt(2)*t1)*wnd
p.figure(5,figsize=(8,8))
p.plot(t1,y,'bo',markeredgecolor='black', markeredgewidth=0.5)
p.plot(t2,y,'ro',markeredgecolor='black', markeredgewidth=0.5)
p.plot(t3,y,'ro',markeredgecolor='black', markeredgewidth=0.5)
p.ylabel("$y$",size=16)
p.xlabel("$t$",size=16)
p.title(r"$\sin\left(\sqrt{2}t\right)\times w(t)$ with $t$ wrapping every $2\pi$ ")
p.grid(True)
p.savefig("fig10-5.png")
p.show()

# Example 6

t = p.linspace(-PI,PI,65);t=t[:-1]
dt = t[1]-t[0]; fmax=1/dt
n = p.arange(64)
wnd = p.fftshift(0.54+0.46*p.cos(2*PI*n/63))
y = p.sin(p.sqrt(2)*t)*wnd
y[0] = 0 # the sample corresponding to -tmax should be set zeroo
y = p.fftshift(y) # make y start with y(t=0)
Y = p.fftshift(p.fft(y))/64.0
w = p.linspace(-PI*fmax,PI*fmax,65);w = w[:-1]
p.figure(6,figsize=(8,8))
p.subplot(2,1,1)
p.plot(w,abs(Y),'b',w,abs(Y),'bo',lw=2)
p.xlim([-8,8])
p.ylabel(r"$|Y|$",size=16)
p.title(r"Spectrum of $\sin\left(\sqrt{2}t\right)\times w(t)$")
p.grid(True)
p.subplot(2,1,2)
p.plot(w,p.angle(Y),'ro',lw=2)
p.xlim([-8,8])
p.ylabel(r"Phase of $Y$",size=16)
p.xlabel(r"$\omega$",size=16)
p.grid(True)
p.savefig("fig10-6.png")
p.show()

# Example 7

t = p.linspace(-4*PI,4*PI,257);t=t[:-1]
dt = t[1]-t[0]; fmax=1/dt
n = p.arange(256)
wnd = p.fftshift(0.54+0.46*p.cos(2*PI*n/255))
y = p.sin(p.sqrt(2)*t)*wnd
y[0] = 0 # the sample corresponding to -tmax should be set zeroo
y = p.fftshift(y) # make y start with y(t=0)
Y = p.fftshift(p.fft(y))/256.0
w = p.linspace(-PI*fmax,PI*fmax,257);w = w[:-1]
p.figure(7,figsize=(8,8))
p.subplot(2,1,1)
p.plot(w,abs(Y),'b',w,abs(Y),'bo',lw=2)
p.xlim([-4,4])
p.ylabel(r"$|Y|$",size=16)
p.title(r"Spectrum of $\sin\left(\sqrt{2}t\right)\times w(t)$")
p.grid(True)
p.subplot(2,1,2)
p.plot(w,p.angle(Y),'ro',lw=2)
p.xlim([-4,4])
p.ylabel(r"Phase of $Y$",size=16)
p.xlabel(r"$\omega$",size=16)
p.grid(True)
p.savefig("fig10-6a.png")
p.show()

# Question 2

# Without Hamming Window
t = p.linspace(-4*PI,4*PI,257);t=t[:-1]
dt = t[1]-t[0]; fmax=1/dt
n = p.arange(256)
y = (p.cos(0.86*t))**3
y[0] = 0 
y = p.fftshift(y) 
Y = p.fftshift(p.fft(y))/256.0
w = p.linspace(-PI*fmax,PI*fmax,257);w = w[:-1]
p.figure(8,figsize=(8,8))
p.subplot(2,1,1)
p.plot(w,abs(Y),'b',w,abs(Y),'bo',lw=2)
p.xlim([-4,4])
p.ylabel(r"$|Y|$",size=16)
p.title(r'Spectrum of $\cos^3(0.86t)$ without hamming window')
p.grid(True)
p.subplot(2,1,2)
p.plot(w,p.angle(Y),'ro',lw=2)
p.xlim([-4,4])
p.ylabel(r"Phase of $Y$",size=16)
p.xlabel(r"$\omega$",size=16)
p.grid(True)
p.savefig("fig10-7a.png")
p.show()

# With Hamming Window
t = p.linspace(-4*PI,4*PI,257);t=t[:-1]
dt = t[1]-t[0]; fmax=1/dt
n = p.arange(256)
wnd = p.fftshift(0.54+0.46*p.cos(2*PI*n/255))
y = (p.cos(0.86*t))**3 * wnd
y[0] = 0 
y = p.fftshift(y) 
Y = p.fftshift(p.fft(y))/256.0
w = p.linspace(-PI*fmax,PI*fmax,257);w = w[:-1]
p.figure(9,figsize=(8,8))
p.subplot(2,1,1)
p.plot(w,abs(Y),'b',w,abs(Y),'bo',lw=2)
p.xlim([-4,4])
p.ylabel(r"$|Y|$",size=16)
p.title(r'Spectrum of $\cos^3(0.86t)$ with hamming window')
p.grid(True)
p.subplot(2,1,2)
p.plot(w,p.angle(Y),'ro',lw=2)
p.xlim([-4,4])
p.ylabel(r"Phase of $Y$",size=16)
p.xlabel(r"$\omega$",size=16)
p.grid(True)
p.savefig("fig10-7b.png")
p.show()

# Question 3 and Question 4
w0 = 1.42
delta = 0.6
t = p.linspace(-PI,PI,129)[:-1]
data = p.cos(w0*t+delta)
data_noise = p.cos(w0*t+delta) + 0.1*p.randn(128)
def WandD(Y, w, k):
    ii = p.where(w>=0)
    avgw = p.sum((abs(Y[ii][:k]))*abs(w[ii][:k]))/p.sum(abs(Y[ii][:k]))
    pp = p.argmax(abs(Y[ii]))
    delta_est = p.angle(Y[ii])[pp]
    return avgw, delta_est


dt = t[1]-t[0]; fmax=1.0/dt
n = p.arange(128)
wnd = p.fftshift(0.54+0.46*p.cos(2*PI*n/127.0))
y = data * wnd
y_noise = data_noise * wnd
y = p.fftshift(y)
y_noise = p.fftshift(y_noise)
Y = p.fftshift(p.fft(y))/128.0
Y_noise = p.fftshift(p.fft(y_noise))/128
w= p.linspace(-PI*fmax,PI*fmax,129)[:-1]

for k in list(range(60))[1:]:
    pres_w, pres_delta = WandD(Y, w, k)
    pres_w_noise, pres_delta_noise = WandD(Y_noise, w, k)
    data_pred = p.cos(pres_w*t+pres_delta)
    data_noise_pred = p.cos(pres_w_noise*t+pres_delta_noise)
    error = max(abs(data_pred-data))
    error_noise = max(abs(data_noise_pred-data_noise))
    if k == 1:
        minerror = error
        minerror_noise = error_noise
    if error<minerror and k>1:
        minerror = error
        est_w = pres_w
        est_delta = pres_delta
    if error_noise<minerror_noise and k>1:
        minerror_noise = error_noise
        est_w_noise = pres_w_noise
        est_delta_noise = pres_delta_noise

print('Question 3 (Without Noise) :')
print("w_estimate: {} delta_estimate: {} ".format(est_w,est_delta))
print('Question 4 (With Noise) :')
print('w_estimate: {} delta_estimate: {} '.format(est_w_noise, est_delta_noise))

# Question 5

# Without Hamming Window
t = p.linspace(-PI,PI,1025);t=t[:-1]
dt = t[1]-t[0]; fmax=1/dt
n = p.arange(1024)
y = p.cos(16*(1.5+t/2/PI)*t) 
y = p.fftshift(y)
Y = p.fftshift(p.fft(y))/1024.0
w = p.linspace(-PI*fmax,PI*fmax,1025);w = w[:-1]
p.figure(10,figsize=(8,8))
p.subplot(2,1,1)
p.plot(w,abs(Y),'b', w,abs(Y),'bo',lw=0.75,markersize=3)
p.xlim([-100,100])
p.ylabel(r"$|Y|$",size=16)
p.title(r'Spectrum of the Chirped Signal $\cos(16(1.5 + t/2\pi)t)$ without hamming window')
p.grid(True)
p.subplot(2,1,2)
p.plot(w,p.angle(Y),'r',w,p.angle(Y),'ro',lw=0.75,markersize=3)
p.xlim([-100,100])
p.ylim([-4,4])
p.ylabel(r"Phase of $Y$",size=16)
p.xlabel(r"$\omega$",size=16)
p.grid(True)
p.savefig("fig10-9a.png")
p.show()

# With Hamming Window
t = p.linspace(-PI,PI,1025);t=t[:-1]
dt = t[1]-t[0]; fmax=1/dt
n = p.arange(1024)
wnd = p.fftshift(0.54+0.46*p.cos(2*PI*n/1023.0))
y = p.cos(16*(1.5+t/2/PI)*t) * wnd
y = p.fftshift(y)
Y = p.fftshift(p.fft(y))/1024.0
w = p.linspace(-PI*fmax,PI*fmax,1025);w = w[:-1]
p.figure(11,figsize=(8,8))
p.subplot(2,1,1)
p.plot(w,abs(Y),'b', w,abs(Y),'bo',lw=0.75,markersize=3)
p.xlim([-100,100])
p.ylabel(r"$|Y|$",size=16)
p.title(r'Spectrum of the Chirped Signal $\cos(16(1.5 + t/2\pi)t)$ with hamming window')
p.grid(True)
p.subplot(2,1,2)
p.plot(w,p.angle(Y),'r',w,p.angle(Y),'ro',lw=0.75,markersize=3)
p.xlim([-100,100])
p.ylim([-3,3])
p.ylabel(r"Phase of $Y$",size=16)
p.xlabel(r"$\omega$",size=16)
p.grid(True)
p.savefig("fig10-9b.png")
p.show()

# Question 6

t = p.linspace(-PI, PI, 1025)[:-1]
fmax = 1/(t[1] - t[0])
Y1 = p.zeros((64,16), dtype=complex)
Y1_wnd = p.zeros((64, 16), dtype=complex)
for i in range(16):
    t1 = t[64*i:(i+1)*64]
    w1 = p.linspace(-PI*fmax, PI*fmax, 65)[:-1]
    y1 = p.cos(16*(1.5 + t1/(2*PI))*t1)
    n = p.arange(64)
    wnd = p.fftshift(0.54 + 0.46*p.cos(2*PI*n/63))
    y1_wnd = y1 * wnd
    y1 = p.fftshift(y1)
    Y1[:, i] = p.fftshift(p.fft(y1))/64.0
    y1_wnd = p.fftshift(y1_wnd)
    Y1_wnd[:, i] = p.fftshift(p.fft(y1_wnd))/64.0
t1 = t[::64]
t1, w1 = p.meshgrid(t1, w1)

fig12 = p.figure(12)
ax = p3.Axes3D(fig12)
surf12 = ax.plot_surface(w1, t1, p.absolute(Y1), cmap=cm.plasma)
fig12.colorbar(surf12)
p.title('DFT Magnitude plot of the Chirped Signal without hamming window')
p.xlabel(r'$\omega$')
p.ylabel(r'$t$')
p.savefig("fig10-10ai.png")

fig13 = p.figure(13)
ax = p3.Axes3D(fig13)
surf13 = ax.plot_surface(w1, t1, p.angle(Y1), cmap=cm.plasma)
fig13.colorbar(surf13)
p.title('DFT Phase plot of the Chirped Signal without hamming window')
p.xlabel(r'$\omega$')
p.ylabel(r'$t$')
p.savefig("fig10-10aii.png")

fig14 = p.figure(14)
ax = p3.Axes3D(fig14)
surf14 = ax.plot_surface(w1, t1, p.absolute(Y1_wnd), cmap=cm.plasma)
fig14.colorbar(surf14)
p.title('DFT Magnitude plot of the Chirped Signal with hamming window')
p.xlabel(r'$\omega$')
p.ylabel(r'$t$')
p.savefig("fig10-10bi.png")

fig15 = p.figure(15)
ax = p3.Axes3D(fig15)
surf15 = ax.plot_surface(w1, t1, p.angle(Y1), cmap=cm.plasma)
fig15.colorbar(surf15)
p.title('DFT Phase plot of the Chirped Signal with hamming window')
p.xlabel(r'$\omega$')
p.ylabel(r'$t$')
p.savefig("fig10-10bii.png")
p.show()
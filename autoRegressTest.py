import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from scipy.signal import hilbert, chirp
import scipy
import pronyAnalysis as pa

Fs = 9600
dt = 1/Fs
t = np.arange(0, 1, dt)
df = Fs/t.size
fVect = np.fft.rfftfreq(t.size, 1/Fs) # np.arange(0, Fs/2-df, df)

dFmax = 0.2
coeff = dFmax/t.size/dt
freq = np.ones((1, t.size)) #50, 60
freq[0, :] = freq[0, :]*100*(1+np.linspace(start=0, stop=dFmax, num=t.size))
#freq[0, 1:480] = freq[0, 1:480]*70 #*(1+np.sin(5*2*np.pi*t)/2)
#freq[0, 481:960] = freq[0, 481:960]*120
#freq[1, :] = freq[1, :]*200
ampl = np.array([1]) #, 0.8
decay = np.array([0]) #, 0.7, 7
phase = np.array([np.pi/4]) #, np.pi/6
signal = np.zeros(t.shape)

for ai in range(len(freq)):
    amplitude = ampl[ai]*np.exp(-decay[ai]*t)
    fullPhase = 2*freq[ai]*np.pi*t + phase[ai]
    comp = amplitude*np.sin(fullPhase)
    signal = signal + comp

signal = chirp(t, freq[0, 0], t[-1], freq[0, -1])
signal = pa.abwgn(signal, SNRdB=10, Fs=Fs, band=(95, 105))[0]
spec = np.fft.rfft(signal)/signal.size
grid = matplotlib.gridspec.GridSpec(2, 1)
fig = plt.figure(figsize=(16, 10))
ax_signal = fig.add_subplot(grid[0])
ax_spectr = fig.add_subplot(grid[1])
ax_phase = ax_spectr.twinx()
ax_signal.plot(t, signal)
ax_spectr.plot(fVect, np.abs(spec))
ax_phase.plot(fVect, np.angle(spec)/np.pi, '1k')
ax_spectr.grid(color = 'r', linestyle = '--')
ax_phase.grid(color = 'k', linestyle = ':')

fig2 = plt.figure(figsize=(16, 10))
matplotlib.pyplot.specgram(signal, Fs = Fs)
#ampl1 = np.exp(-0.5*t)
#phase = 100*np.pi*t
#comp1 = ampl1*np.sin(phase)
order = 4

(windows, timeWin) = pa.windowing(signal, int(np.round(t.size/10)), 75)
f = np.zeros((0, order))
alpha = f
A = f
theta = f
resid = np.zeros((0, 1))
for ai in range(windows.shape[0]):
    (alphaCurr, fCurr, Acurr, thetaCurr, residCurr, coefficient) = pa.chirpProny(windows[ai, :], order, Fs=Fs, step=0.05, limit=0.3)
    alpha = np.vstack((alpha, alphaCurr))
    f = np.vstack((f, fCurr))
    A = np.vstack((A, Acurr))
    theta = np.vstack((theta, thetaCurr))
    resid = np.vstack((resid, residCurr))

'''''
X = np.arange(0, windows[0, :].size, dtype='int')
wresFun = scipy.interpolate.interp1d(X, windows[0, :], kind='cubic', bounds_error=False, fill_value='extrapolate')
dX = np.ones((1, windows[0, :].size))/(1-0.05)
shift = freq[0, np.arange(0, windows[0, :].size, dtype='int')]/freq[0, 0]
dX = np.ones((1, windows[0, :].size))/shift
xNew = np.cumsum(dX)
xNew = xNew[xNew<X.max()]
wres = wresFun(xNew)
fig3 = plt.figure(figsize=(16, 10))
plt.plot(np.arange(0, wres.size, dtype='int'), wres)
plt.plot(np.arange(0, wres.size, dtype='int'), windows[0, :])
'''''

errF = freq[0, timeWin].transpose() - np.abs(f[:, 0])
freqErr = np.sum(errF**2)
predictErr = np.sum(resid)
# ax_signal.plot(t, signal)
ax_spectr.stem(f.transpose(), A, 'c*')
ax_phase.stem(f, theta, 'xm')
print('Alpha')
print(alpha)
print('f')
print(f)
print('A')
print(A)
print('theta')
print(theta)
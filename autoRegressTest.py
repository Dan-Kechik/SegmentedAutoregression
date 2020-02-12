import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from scipy.signal import hilbert, chirp
import scipy
import pronyAnalysis as pa
from matplotlib import cm

plotGraphs=True
Fs = 9600
dt = 1/Fs
t = np.arange(0, 1, dt)
df = Fs/t.size
fVect = np.fft.rfftfreq(t.size, 1/Fs) # np.arange(0, Fs/2-df, df)

dFmax = 0.5
distF = 5
coeff = dFmax/t.size/dt
freq = np.ones((4, t.size)) #50, 60
freq[0, :] = freq[0, :]*100*(1+np.linspace(start=0, stop=dFmax, num=t.size))
freq[1, :] = freq[0, :]+distF
#freq[0, 1:480] = freq[0, 1:480]*70 #*(1+np.sin(5*2*np.pi*t)/2)
#freq[0, 481:960] = freq[0, 481:960]*120
freq[2, :] = freq[2, :]*200
freq[3, :] = freq[2, :]+distF
ampl = np.array([0.7, 0.5, 1, 1]) #
decay = np.array([0, 0, 0, 0]) #, 0.7, 7
phase = np.array([np.pi/4, np.pi/6, np.pi/4, np.pi/6]) #
signal = np.zeros(t.shape)
k = np.zeros_like(decay)

if plotGraphs:
    fig3 = plt.figure()
for ai in (0, 2): #range(len(decay)): #(0, 2): #
    amplitude = ampl[ai]*np.exp(-decay[ai]*t)
    #fullPhase = 2*freq[ai]*np.pi*t + phase[ai]
    comp = amplitude*chirp(t, freq[ai, 0], t[-1], freq[ai, -1]) #*np.sin(fullPhase)
    #comp = pa.abwgn(comp, SNRdB=-7, Fs=Fs, band=(freq[ai, 0]*0.95, freq[ai, -1]*1.05))[0]
    signal = signal + comp
    k[ai] = (freq[ai, -1]-freq[ai, 0])/(t[-1]-t[0])/freq[ai, 0]
    #Plot instantateous frequency
    if plotGraphs:
        analytic_signal = hilbert(comp)
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        instantaneous_frequency = (np.diff(instantaneous_phase) / (2.0 * np.pi) * Fs)
        plt.plot(t[1:], instantaneous_frequency)

# signal = chirp(t, freq[0, 0], t[-1], freq[0, -1])
# signal = pa.abwgn(signal, SNRdB=10, Fs=Fs, band=(95, 105))[0]
signal = pa.awgn(signal, SNRdB=-2)[0]
if plotGraphs:
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
    fig.show()

    fig2 = plt.figure(figsize=(16, 10))
    matplotlib.pyplot.specgram(signal, Fs = Fs)
    fig2.show()

    (representation, t, fVectNew) = pa.DFTbank(signal, rect=1, level=0.2, Fs=Fs, df=1, formFactor=1024)
    #(alpha1, f1, A1, theta1, resid1, coefficient1) = pa.timeSegmentedProny(representation[110, :], order=4, Fs=Fs, percentLength=2.5, percentOverlap=50)
    #(alpha, f, A, theta, resid, coefficient) = pa.timeSegmentedProny(representation[90, :], order=4, Fs=Fs, percentLength=2.5, percentOverlap=50)

    win120 = representation[120, np.arange(2520, 3240, 1, dtype='int')]
    freq120 = freq[0, np.arange(2520, 3240, 1, dtype='int')]
    t120 = t[np.arange(2520, 3240, 1, dtype='int')]
    ax_signal.plot(t120, win120)
    # (alphaR, fr, Ar, thetaR, residR, coefficientR, errorsR) = pa.representProny(signal, representation, fVectNew, Fs=Fs)
    #(alphaCurr, fCurr, Acurr, thetaCurr, residCurr, coefficientCurr) = pa.chirpProny(win120, order=4, Fs=Fs, step=0.05, limit=0.9)
    #(alphaCurr1, fCurr1, Acurr1, thetaCurr1, residCurr1, coefficientCurr1) = pa.chirpProny(signal[0:240], order=4, Fs=Fs, step=0.05, limit=0.9)

    fig4 = plt.figure()
    grid = matplotlib.gridspec.GridSpec(1, 1)
    ax_specWav = fig4.add_subplot(grid[0])
    extent = t[0], t[-1], fVectNew[0], fVectNew[-1]
    ax_specWav.imshow(np.flipud(np.abs(representation)), extent=extent, cmap=plt.get_cmap('binary'))
    fig4.colorbar(cm.ScalarMappable(cmap=plt.get_cmap('binary')), ax=ax_specWav)
    ax_specWav.axis('auto')
    plt.ylim(int(np.min(freq)*0.9), int(np.max(freq)*1.1))
    plt.xlabel('Время, сек')
    plt.ylabel('Частота, Гц')
    # plt.title('Частотно-временное представление сигнала')
    fig4.show()
#ampl1 = np.exp(-0.5*t)
#phase = 100*np.pi*t
#comp1 = ampl1*np.sin(phase)
order = 4

(alphaR, fr, Ar, thetaR, residR, coefficientR, errorsR) = pa.representProny(signal, representation, fVect, Fs=Fs)
pa.segmentedProny(signal, Fs=Fs)

(windows, timeWin) = pa.windowing(signal, int(np.round(t.size/10)), 75)
f = np.zeros((0, order))
alpha = f
A = f
theta = f
resid = np.zeros((0, 1))
for ai in range(windows.shape[0]):
    (alphaCurr, fCurr, Acurr, thetaCurr, residCurr, coefficient) = pa.chirpProny(windows[ai, :], Fs=Fs, step=0.05, limit=0.3)
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
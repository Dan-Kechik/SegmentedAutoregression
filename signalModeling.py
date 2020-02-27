import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from scipy.signal import hilbert, chirp
import scipy
import pronyAnalysis as pa
from matplotlib import cm


def genTime(signal=None, length=None, maxTime=None, Fs=1, dtype=None):
    dt = 1/Fs
    if not length:
        if signal: length = signal.size
    if not maxTime:
        maxTime = dt*length
    t = np.arange(start=0, stop=maxTime, step=dt, dtype=dtype)
    return t


def makeNP(values):
    if not type(values) is np.ndarray:
        values = np.array([values])
    return values


def frequencyModulated(f, t, f0=None, depth=0, phi0=0):
    f = makeNP(f)
    t = makeNP(t)
    dt = np.diff(np.hstack((0, t)))  # Non-uniform sampling interval is considered.
    if not f0 is None:
        if f.size == 1:
            f = np.sin(2*np.pi*f*t)
        f = f0*(1+depth*f)
    phase = np.cumsum(2*np.pi*f*dt)
    signal = np.sin(phase+phi0)
    return signal, f


def AMsign(t, carr, envel, depth=1):
    carr = np.array(carr)
    envel = np.array(envel)
    if carr.size == 1:
        carr = np.sin(2*np.pi*carr*t)
    if envel.size == 1:
        envel = np.sin(2*np.pi*envel*t)
    envel = 1+depth*envel
    signal = envel*carr
    return signal, carr, envel


def expPulses(t, timeSamples, decay, amplitude):
    signal = np.zeros_like(t)
    realTimeSamples = np.zeros_like(timeSamples, dtype='int')
    decay = np.array(decay)
    amplitude = np.array(amplitude)
    if decay.size<timeSamples.size:
        decay = np.full_like(timeSamples, decay)
    if amplitude.size<timeSamples.size:
        amplitude = np.full_like(timeSamples, amplitude)
    for ai in range(timeSamples.size):
        pulse = amplitude[ai]*np.exp(-decay[ai]*t)
        realTimeSamples[ai] = closeInVect(t, timeSamples[ai])[1]
        resid = t.size - realTimeSamples[ai]
        signal[realTimeSamples[ai]:] += pulse[:resid]

    return signal, realTimeSamples


def closeInVect(vect, values):
    values = makeNP(values)
    samples = np.zeros_like(vect, shape=values.shape)
    indexes = np.zeros_like(values, dtype='int')
    for ai in range(indexes.size):
        val = np.array(values[ai])
        indexes[ai] = np.argmin(np.abs(vect - val))
        samples[ai] = vect[indexes[ai]]
    return samples, indexes


def plotSignal(signal, t=None, specKind=None):
    fig = plt.figure()
    if not specKind is None:
        Fs = 1/np.mean(np.diff(t))
        fVect = np.fft.rfftfreq(t.size, 1 / Fs)
        spec = np.fft.rfft(signal) / signal.size
        grid = matplotlib.gridspec.GridSpec(2, 1)
        ax_spectr = fig.add_subplot(grid[1])
        ax_spectr.grid(color='r', linestyle='--')
        ax_spectr.plot(fVect, np.abs(spec))
        if specKind == 'phase':
            ax_phase = ax_spectr.twinx()
            ax_phase.plot(fVect, np.angle(spec) / np.pi, '1k')
            ax_phase.grid(color='k', linestyle=':')
    else:
        grid = matplotlib.gridspec.GridSpec(1, 1)
    ax_signal = fig.add_subplot(grid[0])
    ax_signal.plot(t, signal, color='k')
    fig.show()
    return fig


def plotUnder(t, signal, yLims=None, secondParam=None, secondLim=None, trueVals=None):
    fig = plt.figure()
    fig.show()
    grid = matplotlib.gridspec.GridSpec(len(signal), 1)
    for ai in range(len(signal)):
        ax_curr = fig.add_subplot(grid[ai])
        ax_curr.plot(t, signal[ai], color='k')
        ax_curr.grid(color='k', linestyle=':')
        #ax_curr.set_xlim((0, 1))
        if not trueVals is None:
            if not trueVals[ai] is None:
                ax_curr.plot(t, trueVals[ai], color='b')
                idxs = np.hstack(np.nonzero(signal[ai]))
                print('Param {} RMSE {}'.format(ai, pa.rms(trueVals[ai][100:-100]-signal[ai][100:-100])))
        if not yLims is None:
            ax_curr.set_ylim(yLims[ai])
        if not secondParam is None:
            ax_sec = ax_curr.twinx()
            ax_sec.plot(t, secondParam, '--r')
            ax_sec.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
            if not secondLim is None:
                ax_sec.set_ylim(secondLim)
    return fig


def plotRepresentation(t, representation, fVect, freq):
    fig4 = plt.figure()
    grid = matplotlib.gridspec.GridSpec(1, 1)
    ax_specWav = fig4.add_subplot(grid[0])
    extent = t[0], t[-1], fVect[0], fVect[-1]
    ax_specWav.imshow(np.flipud(np.abs(representation)), extent=extent)
    ax_specWav.axis('auto')
    if not freq is None:
        plt.ylim(int(np.min(freq) * 0.9), int(np.max(freq) * 1.1))
    fig4.show()
    return fig4


def main():
    matplotlib.rc('font', family='Times New Roman')
    Fs = 2000
    ampl = 1
    decay = 15
    t = genTime(maxTime=1, Fs=Fs)
    dFmax=0.2

    # freq = np.ones((1, t.size))
    # freq[0, :] *= 100 * (1 + np.linspace(start=0, stop=dFmax, num=t.size))
    # freq = AMsign(t, np.ones((1, t.size))*100, 5, depth=0.1)[0]
    (FMcomp, freq) = frequencyModulated(5, t, f0=100, depth=0.1, phi0=0)  # chirp(t, freq[0, 0], t[-1], freq[0, -1])
    FMsignal = pa.awgn(FMcomp, SNRdB=10)[0]
    (representation, t0, fVectNew) = pa.DFTbank(FMsignal, rect=2, level=0.2, Fs=Fs, mirrorLen=0.15, df=0.05,
                                                freqLims=(50, 200), formFactor=128)
    tEnd = int(closeInVect(t, 0.15)[1])
    fInd = int(closeInVect(fVectNew, 100)[1])
    fig4 = plotRepresentation(t, representation, fVectNew, None)
    fig100 = plotSignal(representation[fInd, :], t, specKind='amplitude')
    (alpha, f, A, theta, resid, coefficient) = pa.thresholdRepreProny(representation, fVectNew, Fs=Fs, percentLength=2, percentOverlap=75,
                                                                      lowFreq=85, highFreq=120, hold=1.4)
    figFull = plotUnder(t, (FMcomp, alpha, f), yLims=[(-1.2, 1.2), (-30, 30), (85, 125)], trueVals=(None, None, freq), secondParam=resid, secondLim=None)  # (0, 2*10 ** -6)

    ams = AMsign(t, FMcomp, 5, depth=0.25)
    signal = pa.awgn(ams[0], SNRdB=60)[0]
    # figAM = plotSignal(signal, t, specKind='amplitude')
    (representation, t0, fVectNew) = pa.DFTbank(signal, rect=2, level=0.2, Fs=Fs, mirrorLen=0.15, df=1,
                                                freqLims=(50, 200), formFactor=128)
    fig4 = plotRepresentation(t, representation, fVectNew, None)
    fInd = int(closeInVect(fVectNew, 105)[1])
    fig100 = plotSignal(representation[fInd, :], t, specKind='amplitude')
    (alpha, f, A, theta, resid, coefficient) = pa.thresholdRepreProny(representation, fVectNew, Fs=Fs, percentLength=2, percentOverlap=75,
                                                                      lowFreq=80, highFreq=125, hold=1.4)
    figFull = plotUnder(t, (FMcomp, alpha, f), yLims=[(-1.2, 1.2), (-30, 30), (85, 125)], trueVals=(None, None, freq), secondParam=resid, secondLim=None)  # (0, 2*10 ** -6)

    timeSituations = np.arange(start=0, stop=5, step=0.5)
    timeSamples = closeInVect(t, timeSituations)[0]
    envel = expPulses(t, timeSamples, decay, ampl)[0] - np.max(ampl)
    ams = AMsign(t, 100, envel, depth=1)
    signal = pa.awgn(ams[0], SNRdB=-2)[0]
    # fig = plotSignal(signal, t, specKind='amplitude')
    (representation, t0, fVectNew) = pa.DFTbank(signal, rect=2, level=0.2, Fs=Fs, mirrorLen=0.15, df=1,
                                                freqLims=(50, 200), formFactor=128)
    # fig4 = plotRepresentation(t, representation, fVectNew, None)
    (alpha, f, A, theta, resid, coefficient) = pa.thresholdRepreProny(representation, fVectNew, Fs=Fs, periodsNum=10,
                                                                      lowFreq=95, highFreq=105, hold=1.5)
    '''''
    fig3 = plt.figure()
    plt.plot(t, alpha)
    fig2 = plt.figure()
    plt.plot(t, f)
    fig4 = plt.figure()
    plt.plot(t, resid)
    fig4.show()
    '''''
    print(np.mean((f[:int(f.size * 0.85)] - 100) ** 2))
    # figFull = plotUnder(t, (representation[fInd, :], alpha, f), yLims=[(-0.6, 0.6), (12.5, 17.5), (95, 105)], secondParam=resid, secondLim=(0, 7*10**-6))
    # fig100 = plotSignal(representation[fInd, :], t, specKind='amplitude')
    (alpha1, f1, A1, theta1, resid1) = pa.pronyDecomp(representation[fInd, :tEnd], 2, Fs=Fs)
    print(np.sum(resid1.bse ** 2))

    ams = AMsign(t, 100, 5, depth=0.25)
    signal = pa.awgn(ams[0], SNRdB=-2)[0]
    # figAM = plotSignal(signal, t, specKind='amplitude')
    (representation, t0, fVectNew) = pa.DFTbank(signal, rect=2, level=0.2, Fs=Fs, mirrorLen=0.15, df=1,
                                                freqLims=(50, 200), formFactor=1024)
    fig4 = plotRepresentation(t, representation, fVectNew, None)
    fig100 = plotSignal(representation[fInd, :], t, specKind='amplitude')
    (alpha, f, A, theta, resid, coefficient) = pa.thresholdRepreProny(representation, fVectNew, Fs=Fs, percentLength=5, percentOverlap=75,
                                                                      lowFreq=95, highFreq=105, hold=1.4)
    figFull = plotUnder(t, (representation[fInd, :], alpha, f), yLims=[(-2, 2), (-20, 20), (95, 105)],
                        secondParam=resid, secondLim=None) # (0, 2*10 ** -6)
    pass


if __name__ == "__main__":
    main()

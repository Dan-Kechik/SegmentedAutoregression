import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from scipy.signal import hilbert, chirp
import scipy
import pronyAnalysis as pa
from matplotlib import cm
import copy


def genTime(signal=None, length=None, maxTime=None, Fs=1, dtype=None):
    dt = 1/Fs
    if not length:
        if signal: length = signal.size
    if not maxTime:
        maxTime = dt*length
    t = np.arange(start=0, stop=maxTime, step=dt, dtype=dtype)
    return t


def validateTrack(f):
    dff = np.abs(np.diff(f))  # Frequency jumps.
    # Indexes of significant frequency jumps.
    nz = np.nonzero(dff > 7)
    nz = np.hstack((0, nz[0], f.size))
    # Windows lengths.
    dnz = np.diff(nz)
    validWinds = np.hstack(np.nonzero(dnz > 0.3 * f.size))
    fNew = np.zeros_like(f)
    for di in range(validWinds.size):
        currIdxs = nz[validWinds[di]:validWinds[di] + 2]
        fNew[np.arange(start=currIdxs[0] + 1, stop=currIdxs[1], step=1, dtype='int')] = f[
            np.arange(start=currIdxs[0] + 1, stop=currIdxs[1], step=1, dtype='int')]
    validIdxs = np.hstack(np.nonzero(fNew))
    return validIdxs


def makeNP(values, dtype=None):
    if not type(values) is np.ndarray:
        values = np.array([values], dtype=dtype)
    return values


def check_like(a, ref=None, dtype=None, order='K', subok=True, shape=None):
    if not ref is None:
        a = makeNP(a)
        ref = makeNP(ref)
        if a.size == 1:
            a = np.full_like(ref, a, dtype=dtype, order=order, subok=subok, shape=shape)
    return a


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
    signal = signal.squeeze()
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


def plotUnder(t, signal, yLims=None, xLims=None, secondParam=None, secondLim=None, labels=None, secondLabel=None, xlabel=None, ylabel=None, secLabel=None):
    t = makeNP(t, dtype='float64')
    linestyles = ('-', '--', ':', '-.')
    fig = plt.figure()
    fig.show()
    grid = matplotlib.gridspec.GridSpec(len(signal), 1)
    for ai in range(len(signal)):
        if signal[ai] is None:
            continue
        ax_curr = fig.add_subplot(grid[ai])
        lines = []
        labelCurr = labels[ai] if type(labels) in [tuple, list] else labels
        if type(signal[ai]) in [tuple, list]:
            for bi in range(len(signal[ai])):
                if signal[ai][bi] is None:
                    continue
                labelSub = labelCurr[bi] if type(labelCurr) in [tuple, list] else labelCurr
                lines.extend(ax_curr.plot(t, check_like(signal[ai][bi], t), color='k', linestyle=linestyles[bi], label=labelSub))
        else:
            lines.extend(ax_curr.plot(t, check_like(signal[ai], t), color='k', label=labelCurr))
        ax_curr.grid(color='k', linestyle=':')
        if not xLims is None:
            ax_curr.set_xlim(xLims[ai])
        if not yLims is None:
            ax_curr.set_ylim(yLims[ai])
        if not xlabel is None:
            ax_curr.set_xlabel(xlabel)
        labelCurr = ylabel[ai] if type(ylabel) in [tuple, list] else ylabel
        if not labelCurr is None:
            ax_curr.set_ylabel(labelCurr)
        if not secondParam is None:
            ax_sec = ax_curr.twinx()
            lines.extend(ax_sec.plot(t, check_like(secondParam, t), '-.r', label=secondLabel))
            #ax_curr.plot(np.nan, np.nan, '-.r', label=secondLabel)  # Plot to add to the common legend.
            ax_sec.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
            #ax_sec.legend()
            if not xLims is None:
                ax_sec.set_xlim(xLims[ai])
            if not secondLim is None:
                ax_sec.set_ylim(secondLim)
            if not secLabel is None:
                ax_sec.set_ylabel(secLabel)
        labs = [l.get_label() for l in lines]
        ax_curr.legend(lines, labs)
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


def pulsesParamsEst(signal, **kwargs):
    Fs = kwargs.get('Fs', 1)
    (representation, t0, fVectNew) = pa.DFTbank(signal, rect=2, level=0.2, Fs=Fs, mirrorLen=0.15, df=1,
                                                freqLims=(50, 200), formFactor=128)
    (alpha, f, A, theta, resid, coefficient) = pa.thresholdRepreProny(representation, fVectNew, Fs=Fs, periodsNum=10,
        lowFreq=75, highFreq=145, hold=1.4, errorThresh=1, dummyVal=np.nan) #, diffHold=1
    tempDecay = np.zeros_like(alpha)+alpha
    tempDecay[np.isnan(alpha) == True] = -100
    timeSamples = np.nonzero(np.diff(np.hstack((-100, tempDecay)))>100)[0]
    if kwargs.get('t') and kwargs.get('plotGraphs', 0) == 2:
        t = makeNP(kwargs.get('t', 1))
        t = genTime(maxTime=t, Fs=Fs) if t.size == 1 else t
        fInd = int(closeInVect(fVectNew, np.nanmedian(f))[1])
        plotUnder(t, (representation[fInd, :], (alpha, kwargs.get('decay')),  (f, kwargs.get('carrier'))), secondParam=resid,
                  labels=('Signal', ('Estimated decay', 'Decay'), ('Estimated frequency', 'frequency')), secondLabel='Error',
                  secondLim=(0, np.nanmedian(resid)*1.1), xlabel='Time, sec', ylabel=('Filtered signal', 'Exponential decay', 'Frequency'), secLabel='Approximation error',
                  yLims=(None, (np.nanmedian(alpha)-np.nanstd(alpha)*3, np.nanmedian(alpha)+np.nanstd(alpha)*3), None))
    return alpha, f, A, theta, resid, timeSamples, coefficient, representation, fVectNew


def modelPulses(**kwargs):
    ampl = makeNP(kwargs.get('amplitude', 1))
    decay = makeNP(kwargs.get('decay', 0))
    t = makeNP(kwargs.get('t', 1))
    Fs = kwargs.get('Fs', 1)
    t = genTime(maxTime=t, Fs=Fs) if t.size == 1 else t
    timeSituations = np.arange(start=kwargs.get('start', t[0]), stop=kwargs.get('stop', t[-1]), step=kwargs.get('step', (t[-1]-t[0])/10))
    timeSamples = closeInVect(t, timeSituations)[0]
    envel = expPulses(t, timeSamples, decay, ampl)[0] - np.max(ampl)
    ams = AMsign(t, kwargs.get('carrier', 0.1*Fs), envel, depth=1)
    signalTupl = pa.awgn(ams[0], SNRdB=makeNP(kwargs.get('SNR', np.inf)))
    return (signalTupl[0],) + (timeSamples, t)


def pulsesTest(**kwargs):
    SNRs = makeNP(kwargs.get('SNRvals', np.inf)).reshape(-1, 1)
    errAlph = np.zeros_like(SNRs, dtype='float64')
    errF = np.zeros_like(SNRs, dtype='float64')
    errT = np.zeros_like(SNRs, dtype='float64')
    resids = np.zeros_like(SNRs, dtype='float64')
    for ai, SNR in enumerate(kwargs.get('SNRvals', [np.inf])):
        for bi in range(kwargs.get('experiences', 1)):
            kwargs.update({'SNR':SNR})
            (signal, timeSamples, t) = modelPulses(**kwargs)
            (alpha, f, A, theta, res, timeSamplesIdsEst) = pulsesParamsEst(signal, **kwargs)[0:6]
            errAlph[ai] += pa.rms(alpha-makeNP(kwargs.get('decay', 0)))
            errF[ai] += pa.rms(f - makeNP(kwargs.get('carrier', 0)))
            resids[ai] += np.nanmean(res[res<np.inf])
            timeSamplesTemp = closeInVect(timeSamples, t[timeSamplesIdsEst])[0]  # Consider missed pulses.
            errT += pa.rms(t[timeSamplesIdsEst] - timeSamplesTemp)
        errAlph /= kwargs.get('experiences', 1)
        errF /= kwargs.get('experiences', 1)
        errT /= kwargs.get('experiences', 1)
        resids /= kwargs.get('experiences', 1)
    plotUnder(SNRs, (errAlph, errF, errT), secondParam=resids, ylabel=('Decay RMSE', 'Frequency RMSE', 'Time RMSE'),
              secLabel='Approximation error', xlabel='SNR, dB', labels='Estimation error', secondLabel='Approximation error')


def modelModulated(**kwargs):
    t = makeNP(kwargs.get('t', 1))
    Fs = kwargs.get('Fs', 1)
    t = genTime(maxTime=t, Fs=Fs) if t.size == 1 else t
    (FMcomp, freq) = frequencyModulated(kwargs.get('FMfreq', 0), t, f0=kwargs.get('carrier', 0.1*Fs), depth=kwargs.get('FMdepth', 0), phi0=0)
    ams = AMsign(t, FMcomp, kwargs.get('AMfreq', 0), depth=kwargs.get('AMdepth', 0))
    signalTupl = pa.awgn(ams[0], SNRdB=makeNP(kwargs.get('SNR', np.inf)))
    return (signalTupl[0], t, freq)


def modParamsEst(signal, **kwargs):
    Fs = kwargs.get('Fs', 1)
    (lowFreq, highFreq) = kwargs.get('roughFreqs', (70, 150))
    (representation, t0, fVectNew) = pa.DFTbank(signal, rect=2, level=0.2, Fs=Fs, mirrorLen=0.15, df=0.05,
                                                freqLims=(50, 200), formFactor=128)
    # fVN = np.array([80, 85, 90, 95, 100, 105, 110, 125, 135])
    # sc = pa.scalogramFromRepresentation(representation)
    (alpha, f, A, theta, resid, coefficient) = pa.thresholdRepreProny(representation, fVectNew, Fs=Fs, percentLength=2,
          percentOverlap=75, lowFreq=lowFreq, highFreq=highFreq, hold=1.4, dummyVal=np.nan)
    if kwargs.get('t') and kwargs.get('plotGraphs', 0) == 2:
        t = makeNP(kwargs.get('t', 1))
        t = genTime(maxTime=t, Fs=Fs) if t.size == 1 else t
        fInd = int(closeInVect(fVectNew, np.nanmedian(f))[1])
        plotUnder(t, (signal, alpha,  (f, kwargs.get('freq'))), secondParam=resid,
                  labels=('Signal', 'Estimated decay', ('Estimated frequency', 'Real frequency')), secondLabel='Error',
                  secondLim=(0, np.nanmedian(resid)*1.1), xlabel='Time, sec', ylabel=('Modeled signal', 'Exponential decay', 'Frequency'), secLabel='Approximation error',
                  yLims=(None, (np.nanmedian(alpha)-np.nanstd(alpha)*3, np.nanmedian(alpha)+np.nanstd(alpha)*3), None))
    return alpha, f, A, theta, resid, coefficient, representation, fVectNew


def modTest(**kwargs):
    SNRs = makeNP(kwargs.get('SNRvals', np.inf)).reshape(-1, 1)
    exper = kwargs.get('experiences', 1)
    #errAlph = np.zeros_like(SNRs, dtype='float64')
    errFvect = []
    errHvect = []
    errRvect = []
    errF = np.zeros_like(SNRs, dtype='float64')
    errH = np.zeros_like(SNRs, dtype='float64')
    errR = np.zeros_like(SNRs, dtype='float64')
    resids = np.zeros_like(SNRs, dtype='float64')
    residMeans = []
    residMeds = []
    residSums = []
    residNoiseMeans = []
    residNoiseMeds = []
    residNoiseSums = []
    detectRate = np.zeros_like(SNRs, dtype='float64')
    detectLen = np.zeros_like(SNRs, dtype='float64')
    detectHilRate = np.zeros_like(SNRs, dtype='float64')
    detectHilLen = np.zeros_like(SNRs, dtype='float64')
    detectRateRepr = np.zeros_like(SNRs, dtype='float64')
    detectLenRepr = np.zeros_like(SNRs, dtype='float64')
    for ai, SNR in enumerate(kwargs.get('SNRvals', [np.inf])):
        errFvect.append([])
        errHvect.append([])
        errRvect.append([])
        residMeans.append(np.zeros_like(np.arange(exper), dtype='float64'))
        residMeds.append(np.zeros_like(np.arange(exper), dtype='float64'))
        residSums.append(np.zeros_like(np.arange(exper), dtype='float64'))
        residNoiseMeans.append(np.zeros_like(np.arange(exper), dtype='float64'))
        residNoiseMeds.append(np.zeros_like(np.arange(exper), dtype='float64'))
        residNoiseSums.append(np.zeros_like(np.arange(exper), dtype='float64'))
        kwCopy = [copy.deepcopy(kwargs) for i in range(exper)]
        for bi in range(exper):
            # //Get signal//
            kwargs.update({'SNR': SNR})
            (signal, t, freq) = modelModulated(**kwargs)
            # //Get Prony track//
            kwargs.update({'freq': freq, 'roughFreqs': (70, 150), 'iterations': 1, 'formFactor': (64, 128)})
            #(alpha, f, A, theta, res) = modParamsEst(signal, **kwargs)[0:5]
            (alpha, f, A, theta, res, coefficient, representation, fVectNew) = pa.pronyParamsEst(signal, **kwargs)
            kwargs.update({'track': f, 'hold': 8, 'truthVals': freq})
            VT = pa.validateTrack(**kwargs)
            if len(VT) > 0:
                detectRate[ai] += 1
                detectLen[ai] += np.array(VT).size/f.size
            # //Get Hilbert track//
            hilFreq = pa.hilbertTrack(representation, fVectNew, kwargs.get('Fs', 1), 100)
            kwargs.update({'track': hilFreq, 'hold': 8, 'truthVals': freq})
            VT = pa.validateTrack(**kwargs)
            if len(VT) > 0:
                detectHilRate[ai] += 1
                detectHilLen[ai] += np.array(VT).size / hilFreq.size
            # //Get representation track//
            kwargs.update({'formFactor': 1024})
            (repFreq, sc, peaks) = pa.scalogramFinding(signal=signal, rect=2, level=0.2, mirrorLen=0.15, df=0.05,
                                freqLims=(50, 200), **kwargs)  # Fs and plot enabling.
            kwargs.update({'track': repFreq, 'hold': 8, 'truthVals': freq})
            VT = pa.validateTrack(**kwargs)
            if len(VT) > 0:
                detectRateRepr[ai] += 1
                detectLenRepr[ai] += np.array(VT).size/repFreq.size
            if kwargs.get('plotGraphs', 0) == 2:
                plotUnder(t, ((freq, f, repFreq),), labels=(('Real frequency', 'Prony frequency', 'Spectrogram frequency'),))
            #errAlph[ai] += pa.rms(alpha-makeNP(kwargs.get('decay', 0)))
            idx = np.isnan(f) == False
            errF[ai] += pa.rms(f[idx] - freq[idx])
            errFvect[ai].append(f[idx] - freq[idx])
            idx = np.isnan(hilFreq) == False
            errH[ai] += pa.rms(hilFreq[idx] - freq[idx])
            errHvect[ai].append(hilFreq[idx] - freq[idx])
            errR[ai] += pa.rms(repFreq - freq)
            errRvect[ai].append(repFreq - freq)
            resids[ai] += np.nanmean(res[idx])
            residMeans[ai][bi] = np.nanmean(res)
            residMeds[ai][bi] = np.nanmedian(res)
            residSums[ai][bi] = np.nansum(res)
            kwargs.update({'freq': freq, 'roughFreqs': (70, 150), 'iterations': 1, 'formFactor': (64, 128)})
            noise = np.random.normal(loc=0.0, scale=np.sum(signal**2)/signal.size, size=signal.shape)
            (alphaN, fN, AN, thetaN, resN, coefficientN, representationN, fVectNewN) = pa.pronyParamsEst(noise, **kwargs)
            residNoiseMeans[ai][bi] = np.nanmean(resN)
            residNoiseMeds[ai][bi] = np.nanmedian(resN)
            residNoiseSums[ai][bi] = np.nansum(resN)
            #timeSamplesTemp = closeInVect(timeSamples, t[timeSamplesIdsEst])[0]  # Consider missed pulses.
            #errT += pa.rms(t[timeSamplesIdsEst] - timeSamplesTemp)
            if kwargs.get('plotGraphs', 0) == 2:
                plotUnder(t, (signal, (f, freq), (repFreq, freq), (hilFreq, freq)),
                             labels=('Signal', ('Estimated frequency', 'Real frequency'), ('Estimated frequency', 'Real frequency'), ('Estimated frequency', 'Real frequency')),
                             ylabel=('Modeled signal', 'Segmented Prony', 'Wavelet coefficients maximum', 'Hilbert instantaneous frequency'))
            pass
    pass
    #errAlph /= kwargs.get('experiences', 1)
    errF /= kwargs.get('experiences', 1)
    errH /= kwargs.get('experiences', 1)
    errR /= kwargs.get('experiences', 1)
    #errT /= kwargs.get('experiences', 1)
    resids /= kwargs.get('experiences', 1)
    detectLen /= kwargs.get('experiences', 1)
    detectRate /= kwargs.get('experiences', 1)
    detectLenRepr /= kwargs.get('experiences', 1)
    detectRateRepr /= kwargs.get('experiences', 1)
    plotUnder(SNRs, (errF, errR), secondParam=resids, ylabel=('Prony frequency RMSE', 'Spectrogram frequency RMSE'),
              secLabel='Approximation error', xlabel='SNR, dB', labels='Estimation error', secondLabel='Approximation error')


def main():
    matplotlib.rc('font', family='Times New Roman', size=12)
    Fs = 2000
    ampl = 1
    decay = 10
    t = genTime(maxTime=1, Fs=Fs)
    dFmax=0.2
    plotGraphs=0

    #modTest(Fs=Fs, t=1, SNRvals=(0,), carrier=100, FMfreq=5, FMdepth=0.1, AMfreq=4, AMdepth=0.25, plotGraphs=plotGraphs, experiences=1)  # 6, -12
    modTest(Fs=Fs, t=1, SNRvals=np.hstack((np.arange(4, -5, -1), np.arange(-5, -7.5, -0.5), np.arange(-8, -16, -2))),
            carrier=100, FMfreq=5, FMdepth=0.1, AMfreq=5, AMdepth=0, plotGraphs=plotGraphs, experiences=100)  # 6, -12
    pulsesTest(Fs=Fs, decay=decay, t=5, SNRvals=np.arange(0, -12, -2), carrier=100, plotGraphs=plotGraphs, experiences=5)

    # freq = np.ones((1, t.size))
    # freq[0, :] *= 100 * (1 + np.linspace(start=0, stop=dFmax, num=t.size))
    # freq = AMsign(t, np.ones((1, t.size))*100, 5, depth=0.1)[0]
    (FMcomp, freq) = frequencyModulated(5, t, f0=100, depth=0.1, phi0=0)  # chirp(t, freq[0, 0], t[-1], freq[0, -1])
    FMsignal = pa.awgn(FMcomp, SNRdB=60)[0]
    (representation, t0, fVectNew) = pa.DFTbank(FMsignal, rect=2, level=0.2, Fs=Fs, mirrorLen=0.15, df=0.05,
                                                freqLims=(50, 200), formFactor=128)
    tEnd = int(closeInVect(t, 0.15)[1])
    fInd = int(closeInVect(fVectNew, 100)[1])
    fig4 = plotRepresentation(t, representation, fVectNew, None)
    fig100 = plotSignal(representation[fInd, :], t, specKind='amplitude')
    (alpha, f, A, theta, resid, coefficient) = pa.thresholdRepreProny(representation, fVectNew, Fs=Fs, percentLength=2, percentOverlap=75,
                                                                      lowFreq=85, highFreq=120, hold=1.4)
    figFull = plotUnder(t, (FMcomp, alpha, (f, freq)), yLims=[(-1.2, 1.2), (-30, 30), (85, 125)], labels=(None, 'decay', ('estimated frequency', 'initial frequency')), secondParam=resid, secondLim=None)  # (0, 2*10 ** -6)

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
    figFull = plotUnder(t, (FMcomp, alpha, (f, freq)), yLims=[(-1.2, 1.2), (-30, 30), (85, 125)], secondParam=resid, secondLim=None)  # (0, 2*10 ** -6)

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

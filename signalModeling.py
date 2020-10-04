import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from scipy.signal import hilbert, chirp
import scipy
import pronyAnalysis as pa
from matplotlib import cm
import copy
import dill
from multiprocessing import Pool


def genTime(signal=None, length=None, maxTime=None, Fs=1, dtype=None):
    dt = 1/Fs
    if not length:
        if not signal is None: length = signal.size
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


def check_like(a, ref=None, dtype=None, order='K', subok=True, shape=None, sq=0):
    if not ref is None:
        a = makeNP(a)
        ref = makeNP(ref)
        if a.size == 1:
            a = np.full_like(ref, a, dtype=dtype, order=order, subok=subok, shape=shape)
    if sq:
        a = a.squeeze()
    return a


def frequencyModulated(f, t, f0=None, depth=0, phi0=0, linear=False):
    f = makeNP(f)
    t = makeNP(t)
    dt = np.diff(np.hstack((0, t)))  # Non-uniform sampling interval is considered.
    if linear:
        f = f*t  # Implement formula: f = f0(1+ft), where f0 - initial frequency, f - linear coefficient.
        depth = 1
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


def plotUnder(t, signal, yLims=None, xLims=None, secondParam=None, secondLim=None, labels=None, secondLabel=None, xlabel=None, ylabel=None, secLabel=None, fig=None):
    t = makeNP(t, dtype='float64')
    linestyles = ('-', '--', ':', '-.')
    if fig is None:
        fig = plt.figure()
    fig.show()
    grid = matplotlib.gridspec.GridSpec(len(signal), 1)
    for ai in range(len(signal)):
        if signal[ai] is None:
            continue
        '''''
        if len(fig.axes) < ai+1:
            ax_curr = fig.add_subplot(grid[ai])
        else:
            ax_curr = fig.axes[ai]
        '''
        ax_curr = fig.add_subplot(grid[ai])
        lines = []
        labelCurr = labels[ai] if type(labels) in [tuple, list] else labels
        if type(signal[ai]) in [tuple, list]:
            for bi in range(len(signal[ai])):
                if signal[ai][bi] is None:
                    continue
                labelSub = labelCurr[bi] if type(labelCurr) in [tuple, list] else labelCurr
                lines.extend(ax_curr.plot(t, check_like(signal[ai][bi], t, sq=1), color='k', linestyle=linestyles[bi], label=labelSub))
        else:
            lines.extend(ax_curr.plot(t, check_like(signal[ai], t, sq=1), color='k', label=labelCurr))
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
            lines.extend(ax_sec.plot(t, check_like(secondParam, t, sq=1), '-.r', label=secondLabel))
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
        emptyLables = np.array([labs[i][:5] == '_line' for i in range(len(labs))])
        if not np.all(emptyLables):
            ax_curr.legend(lines, labs)
    return fig


def plotRepresentation(t, representation, fVect, freq=None, spectrum=None):
    fig4 = plt.figure()
    if spectrum is None:
        grid = matplotlib.gridspec.GridSpec(1, 1)
        ax_specWav = fig4.add_subplot(grid[0])
    else:
        ax_frequency = fig4.add_subplot(121)
        ax_frequency.plot(fVect, spectrum)
        ax_specWav = fig4.add_subplot(122)
    extent = t[0], t[-1], fVect[0], fVect[-1]
    ax_specWav.imshow(np.flipud(np.abs(representation)), extent=extent)
    ax_specWav.axis('auto')
    if not freq is None:
        plt.ylim(int(np.min(freq) * 0.9), int(np.max(freq) * 1.1))
    fig4.show()
    return fig4


def pulsesParamsEst(signal, **kwargs):
    Fs = kwargs.get('Fs', 1)
    (lowFreq, highFreq) = kwargs.get('roughFreqs', (75, 145))
    hold = kwargs.get('hold', 0)
    (representation, t0, fVectNew) = pa.DFTbank(signal, rect=2, level=0.2, Fs=Fs, mirrorLen=0.15, df=1,
                                                freqLims=(lowFreq, highFreq), formFactor=kwargs.get('formFactor', 128))
    (alpha, f, A, theta, resid, coefficient) = pa.thresholdRepreProny(representation, fVectNew, Fs=Fs, periodsNum=10,
        lowFreq=lowFreq, highFreq=highFreq, hold=hold, errorThresh=0.8, dummyVal=np.nan) #, diffHold=1
    tempDecay = np.zeros_like(alpha)+alpha
    tempDecay[np.isnan(alpha) == True] = -100
    timeSamples = np.nonzero(np.diff(np.hstack((-100, tempDecay)))>50)[0]
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
    errPer = np.zeros_like(SNRs, dtype='float64')
    resids = np.zeros_like(SNRs, dtype='float64')
    for ai, SNR in enumerate(kwargs.get('SNRvals', [np.inf])):
        kwargs.update({'SNR':SNR})
        (signal, timeSamples, t) = modelPulses(**kwargs)  # Get model parameters for comparison.
        if not kwargs.get('processes') is None:
            kwCopy = [copy.deepcopy(kwargs) for i in range(kwargs.get('experiences', 1))]
            with Pool(processes=kwargs.get('processes')) as pool:
                if kwargs.get('asyncLoop', False):
                    result = []
                    for r in pool.imap_unordered(pulseExperience, kwCopy):
                        result.append(r)
                else:
                    result = pool.map(pulseExperience, kwCopy)
                pool.close()
                pool.join()
        for bi in range(kwargs.get('experiences', 1)):
            if not kwargs.get('processes') is None:
                (alpha, f, A, theta, res, timeSamplesIdsEst) = result[bi]
            else:
                (alpha, f, A, theta, res, timeSamplesIdsEst) = pulseExperience(kwargs)
            errAlph[ai] += pa.rms(alpha-makeNP(kwargs.get('decay', 0)))
            errF[ai] += pa.rms(f - makeNP(kwargs.get('carrier', 0)))
            resids[ai] += np.nanmean(res[res<np.inf])
            timeSamplesTemp = closeInVect(timeSamples, t[timeSamplesIdsEst])[0]  # Consider missed pulses.
            errT[ai] += pa.rms(t[timeSamplesIdsEst] - timeSamplesTemp)
            errPer[ai] += pa.rms(np.mean(np.diff(timeSamples)) - np.mean(np.diff(t[timeSamplesIdsEst])))
        print(kwargs.get('fileName', 'pulseTest')+' {} SNR'.format(SNR))
    errAlph /= kwargs.get('experiences', 1)
    errF /= kwargs.get('experiences', 1)
    errT /= kwargs.get('experiences', 1)
    errPer /= kwargs.get('experiences', 1)
    resids /= kwargs.get('experiences', 1)
    plotUnder(SNRs, (errAlph, errF, errT), ylabel=('Decay RMSE', 'Frequency RMSE', 'Time RMSE'),  # , secondParam=resids
              xlabel='SNR, dB', labels='Estimation error')  # secLabel='Approximation error', , secondLabel='Approximation error'
    import os
    if not (os.path.exists('Out') and os.path.isdir('Out')):
        os.mkdir('Out')
    fName = kwargs.get('fileName')
    if not fName is None:
        if fName == '':
            fName = 'pulsesEstimation.pkl'
        with open(os.path.join('Out', fName), "wb") as f:
            kwSave = {'SNRs': SNRs, 'errAlph': errAlph, 'errF': errF, 'errT': errT, 'errPer': errPer, 'resids': resids}
            dill.dump(kwSave, f)
            f.close()


def pulseExperience(kwargs):
    (signal, timeSamples, t) = modelPulses(**kwargs)
    (alpha, f, A, theta, res, timeSamplesIdsEst) = pulsesParamsEst(signal, **kwargs)[0:6]
    return alpha, f, A, theta, res, timeSamplesIdsEst


def modelModulated(**kwargs):
    t = makeNP(kwargs.get('t', 1))
    Fs = kwargs.get('Fs', 1)
    t = genTime(maxTime=t, Fs=Fs) if t.size == 1 else t
    (FMcomp, freq) = frequencyModulated(kwargs.get('FMfreq', 0), t, f0=kwargs.get('carrier', 0.1*Fs), depth=kwargs.get('FMdepth', 0), phi0=0, linear=kwargs.get('linear', False))
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
    holdMeds = []
    holdMeans = []
    detectRate = np.zeros_like(SNRs, dtype='float64')
    detectLen = np.zeros_like(SNRs, dtype='float64')
    detectHilRate = np.zeros_like(SNRs, dtype='float64')
    detectHilLen = np.zeros_like(SNRs, dtype='float64')
    detectRateRepr = np.zeros_like(SNRs, dtype='float64')
    detectLenRepr = np.zeros_like(SNRs, dtype='float64')
    for ai, SNR in enumerate(kwargs.get('SNRvals', [np.inf])):
        kwargs.update({'SNR': SNR})
        residMeans.append(np.zeros_like(np.arange(exper), dtype='float64'))
        residMeds.append(np.zeros_like(np.arange(exper), dtype='float64'))
        residSums.append(np.zeros_like(np.arange(exper), dtype='float64'))
        residNoiseMeans.append(np.zeros_like(np.arange(exper), dtype='float64'))
        residNoiseMeds.append(np.zeros_like(np.arange(exper), dtype='float64'))
        residNoiseSums.append(np.zeros_like(np.arange(exper), dtype='float64'))
        if not kwargs.get('processes') is None:
            kwCopy = [copy.deepcopy(kwargs) for i in range(exper)]
            with Pool(processes=kwargs.get('processes')) as pool:
                if kwargs.get('asyncLoop', False):
                    result = []
                    for r in pool.imap_unordered(modelExperience, kwCopy):
                        result.append(r)
                else:
                    result = pool.map(modelExperience, kwCopy)
                pool.close()
                pool.join()
        for bi in range(exper):
            if not kwargs.get('processes') is None:
                (errF1, errH1, errR1, resids1, residMeans[ai][bi], residMeds[ai][bi], residNoiseMeans[ai][bi], residNoiseMeds[ai][bi],
                    errFvect1, errHvect1, errRvect1, detectRate1, detectLen1, detectHilRate1, detectHilLen1, detectRateRepr1, detectLenRepr1) = result[bi]
            else:
                (errF1, errH1, errR1, resids1, residMeans[ai][bi], residMeds[ai][bi], residNoiseMeans[ai][bi], residNoiseMeds[ai][bi],
                    errFvect1, errHvect1, errRvect1, detectRate1, detectLen1, detectHilRate1, detectHilLen1, detectRateRepr1, detectLenRepr1) = modelExperience(kwargs)
            errF[ai] += errF1
            errH[ai] += errH1
            errR[ai] += errR1
            resids[ai] += resids1
            detectRate[ai] += detectRate1
            detectLen[ai] += detectLen1
            detectHilRate[ai] += detectHilRate1
            detectHilLen[ai] += detectHilLen1
            detectRateRepr[ai] += detectRateRepr1
            detectLenRepr[ai] += detectLenRepr1
            errFvect.append(errFvect1)
            errHvect.append(errHvect1)
            errRvect.append(errRvect1)
        holdMeans.append(autoThresholding(residNoiseMeans[ai], residMeans[ai]))
        holdMeds.append(autoThresholding(residNoiseMeds[ai], residMeds[ai]))
        print(kwargs.get('fileName', 'modTest')+' {} SNR'.format(SNR))
    pass
    #errAlph /= kwargs.get('experiences', 1)
    errF /= kwargs.get('experiences', 1)
    errH /= kwargs.get('experiences', 1)
    errR /= kwargs.get('experiences', 1)
    #errT /= kwargs.get('experiences', 1)
    resids /= kwargs.get('experiences', 1)
    detectLen /= kwargs.get('experiences', 1)
    detectRate /= kwargs.get('experiences', 1)
    detectHilLen /= kwargs.get('experiences', 1)
    detectHilRate /= kwargs.get('experiences', 1)
    detectLenRepr /= kwargs.get('experiences', 1)
    detectRateRepr /= kwargs.get('experiences', 1)
    # //Evaluate thresholding statistics//
    TP = makeNP([holdMeans[i].get('trueH0') for i in range(len(holdMeans))])
    TN = makeNP([holdMeans[i].get('trueH1') for i in range(len(holdMeans))])
    FP = makeNP([holdMeans[i].get('falseH0') for i in range(len(holdMeans))])
    FN = makeNP([holdMeans[i].get('falseH1') for i in range(len(holdMeans))])
    figMed = plotUnder(SNRs, ((TP, TN, FP + FN),), labels=(('True positive', 'True negative', 'Common risk'),),
                       xlabel='SNR, dB', ylabel=('Probability',))
    plt.title('Errors means')
    TP = makeNP([holdMeds[i].get('trueH0') for i in range(len(holdMeds))])
    TN = makeNP([holdMeds[i].get('trueH1') for i in range(len(holdMeds))])
    FP = makeNP([holdMeds[i].get('falseH0') for i in range(len(holdMeds))])
    FN = makeNP([holdMeds[i].get('falseH1') for i in range(len(holdMeds))])
    figMed = plotUnder(SNRs, ((TP, TN, FP+FN),), labels=(('True positive', 'True negative', 'Common risk'),), xlabel='SNR, dB', ylabel=('Probability',))
    plt.title('Errors meds')
    figFinal = plotUnder(SNRs, (errF, errR), secondParam=resids, ylabel=('Prony frequency RMSE', 'Spectrogram frequency RMSE'),
              secLabel='Approximation error', xlabel='SNR, dB', labels='Estimation error', secondLabel='Approximation error')
    import os
    if not (os.path.exists('Out') and os.path.isdir('Out')):
        os.mkdir('Out')
    fName = kwargs.get('fileName')
    if not fName is None:
        if fName == '':
            fName = 'probEstimation.pkl'
        with open(os.path.join('Out', fName), "wb") as f:
            kwSave = {'SNRs': SNRs, 'errF': errF, 'errH': errH, 'errR': errR, 'resids': resids, 'residMeans': residMeans,
                      'residMeds': residMeds, 'residNoiseMeans': residNoiseMeans, 'residNoiseMeds': residNoiseMeds,
                    'errFvect': errFvect, 'errHvect': errHvect, 'errRvect': errRvect, 'detectRate': detectRate, 'detectLen': detectLen,
                      'detectHilRate': detectHilRate, 'detectHilLen': detectHilLen, 'detectRateRepr': detectRateRepr, 'detectLenRepr': detectLenRepr, 'holdMeans': holdMeans, 'holdMeds': holdMeds}
            dill.dump(kwSave, f)
            f.close()


def autoThresholding(H1samples, H0samples, **kwargs):
    # H0 - less threshold, H1 - greater.
    (cumH0, scaleH0, denH0) = pa.distributLaw(H0samples, scale=None, dummyVal=0)
    (cumH1, scaleH1, denH1) = pa.distributLaw(H1samples, scale=None, dummyVal=0)
    lim = np.array((np.max(scaleH0), np.min(scaleH1)))
    if lim[0]<=lim[1]:
        optimalHold = np.mean(lim)
        limVect = np.array((0,))
        trueH0 = np.array((1,))
        trueH1=np.array((1,))
        falseH0 = np.array((0,))
        falseH1 = np.array((0,))
        idx = 0
    else:
        lim = np.array((np.max(scaleH1), np.min(scaleH0)))
        if len(lim):
            kwargout = {'limVect': np.array([]), 'trueH0': 0, 'trueH1': 0, 'falseH0': 1, 'falseH1': 1, 'optimalHold': np.nan}
            return kwargout
        dH = np.min((scaleH0[1]-scaleH0[0], scaleH1[1]-scaleH1[0]))
        limVect = np.arange(lim[1], lim[0], dH)
        trueH0 = np.zeros_like(limVect, dtype='float64')
        falseH0 = np.zeros_like(limVect, dtype='float64')
        trueH1 = np.zeros_like(limVect, dtype='float64')
        falseH1 = np.zeros_like(limVect, dtype='float64')
        denH0 /= np.sum(denH0)
        denH1 /= np.sum(denH1)
        for ai, hold in enumerate(limVect):
            trueH0[ai] = cumInt(denH0[scaleH0 < hold])
            falseH0[ai] = cumInt(denH1[scaleH1 < hold])
            trueH1[ai] = cumInt(denH1[scaleH1 > hold])
            falseH1[ai] = cumInt(denH0[scaleH0 > hold])
            # trueH0[ai] = np.nonzero(H0samples < hold)[0].size
            # falseH0[ai] = np.nonzero(H1samples < hold)[0].size
            # trueH1[ai] = np.nonzero(H1samples > hold)[0].size
            # falseH1[ai] = np.nonzero(H0samples > hold)[0].size
        # trueH0 /= limVect.size
        # falseH0 /= limVect.size
        # trueH1 /= limVect.size
        # falseH1 /= limVect.size
        commonRisk = falseH1+falseH0
        commonRight = trueH1 + trueH0
        idx = np.argmin(commonRisk)
        optimalHold = limVect[idx]
    kwargout = {'limVect': limVect, 'trueH0': trueH0[idx], 'trueH1': trueH1[idx], 'falseH0': falseH0[idx],
                'falseH1': falseH1[idx], 'optimalHold': optimalHold}
    return kwargout


def cumInt(vals):
    if any(vals):
        return np.cumsum(vals)[-1]
    else:
        return np.array((0,))

def conventionalPronyExperience(kwdict, **kwargs):
    if type(kwdict) is dict:
        kwargs.update(kwdict)
    Fs = kwargs.get('Fs', 1)
    # //Get signal//
    (signal, t, freq) = modelModulated(**kwargs)
    alpha, f, A, theta, resid, coefficient, signalFiltrated, fVect = pa.conventionalProny(signal, **kwargs)


def modelExperience(kwdict, **kwargs):
    if type(kwdict) is dict:
        kwargs.update(kwdict)
    errFvect = []
    errHvect = []
    errRvect = []
    detectRate = 0
    detectLen = 0
    detectHilRate = 0
    detectHilLen = 0
    detectRateRepr = 0
    detectLenRepr = 0
    # //Get signal//
    (signal, t, freq) = modelModulated(**kwargs)
    # //Get Prony track//
    kwargs.update({'freq': freq, 'roughFreqs': (50, 200), 'iterations': 1, 'formFactor': (64, 128), 'hold': 0})
    # (alpha, f, A, theta, res) = modParamsEst(signal, **kwargs)[0:5]
    if kwargs.get('conventionalProny', False):
        (alpha, f, A, theta, res, coefficient, representation, fVectNew) = pa.conventionalProny(signal, **kwargs)
    else:
        (alpha, f, A, theta, res, coefficient, representation, fVectNew) = pa.pronyParamsEst(signal, **kwargs)
    kwargs.update({'track': 0+f, 'hold': 8})
    VT = pa.validateTrack(**kwargs)
    if len(VT) > 0:
        detectRate += 1
        detectLen += np.array(VT).size / f.size
    # //Get Hilbert track//
    hilFreq = pa.hilbertTrack(representation, fVectNew, kwargs.get('Fs', 1), 100)
    kwargs.update({'track': 0+hilFreq, 'hold': 8})
    VT = pa.validateTrack(**kwargs)
    if len(VT) > 0:
        detectHilRate += 1
        detectHilLen += np.array(VT).size / hilFreq.size
    # //Get representation track//
    kwargs.update({'formFactor': 1024})
    (repFreq, sc, peaks) = pa.scalogramFinding(signal=signal, rect=2, level=0.2, mirrorLen=0.15, df=0.05,
                                               freqLims=(50, 200), **kwargs)  # Fs and plot enabling.
    kwargs.update({'track': 0+repFreq, 'hold': 8})
    VT = pa.validateTrack(**kwargs)
    if len(VT) > 0:
        detectRateRepr += 1
        detectLenRepr += np.array(VT).size / repFreq.size
    if kwargs.get('plotGraphs', 0) == 2:
        plotUnder(t, ((freq, f, repFreq),), labels=(('Real frequency', 'Prony frequency', 'Spectrogram frequency'),))
    # errAlph[ai] += pa.rms(alpha-makeNP(kwargs.get('decay', 0)))
    idx = np.isnan(f) == False
    errF = pa.rms(f[idx] - freq[idx])
    errFvect.append(f[idx] - freq[idx])
    idx = np.isnan(hilFreq) == False
    errH = pa.rms(hilFreq[idx] - freq[idx])
    errHvect.append(hilFreq[idx] - freq[idx])
    errR = pa.rms(repFreq - freq)
    errRvect.append(repFreq - freq)
    resids = np.nanmean(res[idx])
    residMeans = np.nanmean(res)
    residMeds = np.nanmedian(res)
    residSums = np.nansum(res)
    kwargs.update({'freq': freq, 'roughFreqs': (50, 200), 'iterations': 1, 'formFactor': (512, 512)}) #(64, 128)
    noise = np.random.normal(loc=0.0, scale=np.sqrt(np.sum(signal ** 2) / signal.size), size=signal.shape)
    kwargs.update({'hold': 0})
    #(alphaN, fN, AN, thetaN, resN, coefficientN, representationN, fVectNewN) = pa.pronyParamsEst(noise, **kwargs)
    (alphaN, fN, AN, thetaN, resN, coefficientN, representationN, fVectNewN) = (alpha, f, A, theta, res, coefficient, representation, fVectNew)
    residNoiseMeans = np.nanmean(resN)
    residNoiseMeds = np.nanmedian(resN)
    residNoiseSums = np.nansum(resN)
    # timeSamplesTemp = closeInVect(timeSamples, t[timeSamplesIdsEst])[0]  # Consider missed pulses.
    # errT += pa.rms(t[timeSamplesIdsEst] - timeSamplesTemp)
    if kwargs.get('plotGraphs', 0) == 2:
        plotUnder(t, (signal, (f, freq), (repFreq, freq), (hilFreq, freq)),
                  labels=('Signal', ('Estimated frequency', 'Real frequency'), ('Estimated frequency', 'Real frequency'),
                  ('Estimated frequency', 'Real frequency')),
                  ylabel=('Modeled signal', 'Segmented Prony', 'Wavelet coefficients maximum',
                          'Hilbert instantaneous frequency'))
    return errF, errH, errR, resids, residMeans, residMeds, residNoiseMeans, residNoiseMeds, errFvect, errHvect, errRvect, detectRate, detectLen, detectHilRate, detectHilLen, detectRateRepr, detectLenRepr


def main():
    matplotlib.rc('font', family='Times New Roman', size=12)
    Fs = 2000
    ampl = 1
    decay = 10
    t = genTime(maxTime=1, Fs=Fs)
    dFmax=2
    freq = np.linspace(start=0, stop=dFmax, num=t.size)
    plotGraphs=0

   # modTest(Fs=Fs, t=1, SNRvals=np.arange(4, -16.5, -0.5), fileName='linear02AM0.pkl',
            #carrier=100, FMfreq=freq, FMdepth=0.1, AMfreq=5, AMdepth=0.0, plotGraphs=plotGraphs, experiences=102, processes=6, asyncLoop=True)  # 6, -12
    modTest(Fs=Fs, t=1, SNRvals=np.arange(-3, -16.5, -0.5), fileName='AMf5d025.pkl',
            carrier=100, FMfreq=5, FMdepth=0.1, AMfreq=5, AMdepth=0.25, plotGraphs=plotGraphs, experiences=1)  # 6, -12
    return
    pulsesTest(Fs=Fs, decay=decay, t=5, SNRvals=np.arange(-10, -16.5, -0.5), fileName='', carrier=100, plotGraphs=plotGraphs, experiences=3)
    return
    modTest(Fs=Fs, t=1, SNRvals=np.arange(4, -16.5, -0.5), fileName='AMf5d025.pkl',
            carrier=100, FMfreq=5, FMdepth=0.1, AMfreq=5, AMdepth=0.25, plotGraphs=plotGraphs, experiences=102, processes=3, asyncLoop=True)  # 6, -12
    modTest(Fs=Fs, t=1, SNRvals=np.arange(4, -16.5, -0.5), fileName='AMf5d02.pkl',  # np.hstack((np.arange(4, -5, -1), np.arange(-5, -7.5, -0.5), np.arange(-8, -22, -2)))
            carrier=100, FMfreq=5, FMdepth=0.1, AMfreq=5, AMdepth=0.2, plotGraphs=plotGraphs, experiences=102, processes=3, asyncLoop=True)  # 6, -12
    modTest(Fs=Fs, t=1, SNRvals=np.arange(4, -16.5, -0.5), fileName='AMf3d02.pkl',
            carrier=100, FMfreq=5, FMdepth=0.1, AMfreq=3, AMdepth=0.2, plotGraphs=plotGraphs, experiences=102, processes=3, asyncLoop=True)  # 6, -12
    return

    h0Sam = np.random.normal(loc=0.0, scale=1/3, size=(1, 100))  # np.arange(0.1, 1.1, 0.05)
    h1Sam = np.random.normal(loc=1.0, scale=1/2.5, size=(1, 100))  # np.arange(0.9, 2, 0.1)
    autoThresholding(h1Sam, h0Sam)

    from timeit import default_timer as timer
    #modTest(Fs=Fs, t=1, SNRvals=(0,), carrier=100, FMfreq=5, FMdepth=0.1, AMfreq=4, AMdepth=0.25, plotGraphs=plotGraphs, experiences=1)  # 6, -12
    tStart = timer()
    modTest(Fs=Fs, t=1, SNRvals=np.arange(4, 1, -1), fileName='',
            carrier=100, FMfreq=5, FMdepth=0.1, AMfreq=5, AMdepth=0, plotGraphs=plotGraphs, experiences=16, processes=4, asyncLoop=True)  # 6, -12
    elapsed2 = timer() - tStart
    print(('2 async', elapsed2))

    tStart = timer()
    modTest(Fs=Fs, t=1, SNRvals=np.arange(4, 1, -1), fileName='',
            carrier=100, FMfreq=5, FMdepth=0.1, AMfreq=5, AMdepth=0, plotGraphs=plotGraphs, experiences=16, processes=4)  # 6, -12
    elapsed1 = timer() - tStart
    print(('2 map', elapsed1))

    tStart = timer()
    modTest(Fs=Fs, t=1, SNRvals=np.arange(4, 1, -1), fileName='',
            carrier=100, FMfreq=5, FMdepth=0.1, AMfreq=5, AMdepth=0, plotGraphs=plotGraphs, experiences=16)  # 6, -12
    elapsed3 = timer() - tStart
    print(('Consequent', elapsed3))

    modTest(Fs=Fs, t=1, SNRvals=np.hstack((np.arange(4, -5, -1), np.arange(-5, -7.5, -0.5), np.arange(-8, -16, -2))), fileName='',
            carrier=100, FMfreq=5, FMdepth=0.1, AMfreq=5, AMdepth=0, plotGraphs=plotGraphs, experiences=16)  # 6, -12

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

import numpy as np
from statsmodels.tsa.ar_model import AutoReg as AR
import scipy
import matplotlib
from matplotlib import pyplot as plt
from scipy.signal import hilbert
import signalModeling as SM
import random
#from numba import jit


def awgn(signal, SNRdB=None, SNRlin=None, sigPower=None):
    if not SNRlin:
        SNRlin = 0
    if not SNRdB is None:
        SNRlin = SNRlin + 10 ** (SNRdB / 10)
    if not sigPower:
        sigPower = np.sqrt(np.sum(signal**2)/signal.size)
    noisePower = sigPower/SNRlin
    noise = np.random.normal(loc=0.0, scale=noisePower, size=signal.shape)
    return signal+noise, noise, sigPower, noisePower, SNRlin, SNRdB


def abwgn(signal, SNRdB=None, SNRlin=None, sigPower=None, Fs=1, band=None, powerCorrection=False):
    fVect = np.fft.rfftfreq(signal.size, 1 / Fs)
    if not SNRlin:
        SNRlin = 0
    if band:
        freqIndexes = np.argmin(np.abs(fVect-band[0]))
        freqIndexes = np.hstack((freqIndexes, np.argmin(np.abs( fVect-band[1])) ))
        bandReal = fVect[freqIndexes]
        if powerCorrection:
            amplification = (fVect[-1] - fVect[0]) / (bandReal[-1] - bandReal[0])
            SNRlin = SNRlin*amplification
    else:
        bandReal = np.hstack((fVect[0], fVect[-1]))
        amplification = 0
    (noise, sigPower, noisePower, SNRlin, SNRdB) = awgn(signal, SNRdB=SNRdB, SNRlin=SNRlin, sigPower=sigPower)[1:]
    if band:
        noiseSpec = np.fft.rfft(noise)/noise.size
        noiseSpec[np.arange(start=0, stop=freqIndexes[0]-1, step=1, dtype='int')] = 0
        noiseSpec[np.arange(start=freqIndexes[1]+1, stop=noiseSpec.size, step=1, dtype='int')] = 0
        noise = np.fft.irfft(noiseSpec)*noise.size
    return signal+noise, noise, sigPower, noisePower, SNRlin, SNRdB, bandReal


def getSimilarsByNumber(signal, tolerance=0, maxSimilarNumber=np.Inf):
    groupedIndexes=[]
    while not np.all(np.isnan(signal)):
        elemIndex = np.nonzero(not np.isnan(signal))[0]
        elem = signal[elemIndex]
        indexes = np.nonzero(signal-elem <= tolerance)
        groupedIndexes.append(indexes)


def getSimilarsByDistance(signal, tolerance=0, maxSimilarNumber=np.Inf):
    # Select elements by their closeness first, select required number of the closest ones in groups.
    groupedIndexes=[]
    groupedValues=[]
    while not np.all(np.isnan(signal)):
        elemIndex = np.nonzero(not np.isnan(signal))[0]
        elem = signal[elemIndex]
        indexes = np.nonzero(signal-elem <= tolerance)
        groupedValues.append(np.mean(signal[indexes]))
        signal[indexes] = np.nan(indexes.shape)
        groupedIndexes.append(indexes)
    return groupedValues, groupedIndexes


def windowing(x, fftsize=1024, overlap=4, returnShortSignal=True, joinLastSample=False, dtype=None, hop=None):
    if hop is None:
        hop = int(fftsize * (100 - overlap)/100)
    hop = max(hop, 1)
    fftsize = int(fftsize)
    signal = np.array([x[i:i+fftsize] for i in range(0, len(x)-fftsize, hop)], dtype=dtype)
    if returnShortSignal and signal.size == 0:
        signal = np.zeros((1, x.size), dtype=dtype)
        signal[0, :] = x
        if joinLastSample:
            return signal, np.array([0, len(x)-1])
        return signal, 0
    samples = np.array([i for i in range(0, len(x)-fftsize, hop)], dtype='int')
    if joinLastSample:
        samples = np.hstock(samples, samples[-1]+fftsize)
    return signal, samples


def downsampling(signal, Fs, FsNew, overAliasing=1):
    # Make antialiasing using FFT
    spec = np.fft.rfft(signal)
    overAliasing = np.min(overAliasing, 1)
    cutSample = np.floor(FsNew*overAliasing/Fs)
    if cutSample < len(spec):
        spec[cutSample:len(spec)] = 0
    filteratedSignal = np.fft.irfft(spec)
    # Decimate signal


def pronyDecomp(signal, order, epsilon=0, Fs=1):
    dt = 1/Fs
    ar_mod = AR(signal, order)
    ar_res = ar_mod.fit()
    """""
    model : AR model instance
        A reference to the fitted AR model.
    nobs : float
        The number of available observations `nobs` - `k_ar`
    n_totobs : float
        The number of total observations in `endog`. Sometimes `n` in the docs.
    params : array
        The fitted parameters of the model.
    pvalues : array
        The p values associated with the standard errors.
    resid : array
        The residuals of the model. If the model is fit by 'mle' then the
        pre-sample residuals are calculated using fitted values from the Kalman
        Filter.
    roots : array
        The roots of the AR process are the solution to
        (1 - arparams[0]*z - arparams[1]*z**2 -...- arparams[p-1]*z**k_ar) = 0
        Stability requires that the roots in modulus lie outside the unit
        circle.
    """""
    # print(ar_res)

    # Matrix: rows are power of roots 0...N-1, N = length(data); columns are roots 1...p, p - autoregression order.
    Zrows = []
    z = ar_res.roots
    for ai in range(0, len(signal)):
        Zrows.append(z**ai)

    Z = np.array(Zrows)
    ZH = Z.conjugate().transpose()
    # Get matrix and vector from formula: (ZH*Z)h = (ZH*x), Mtx@h=vect
    Mtx = ZH@Z
    x = np.array([signal]).transpose()
    vect = ZH@x
    h = np.linalg.solve(Mtx, vect)  # Solve Mtx@h=vect
    # Search for conjugated roots
    if epsilon > 0:
        (z, groupedIndexes) = getSimilarsByDistance(z, tolerance=epsilon)
        h = [np.mean(h[i]) for i in groupedIndexes]

    re = np.real(h)
    im = np.imag(h)
    alpha = np.log(np.abs(z))/dt
    f = np.arctan(np.imag(z)/np.real(z))/(2*np.pi*dt)
    A = np.abs(h).transpose().squeeze()
    theta = np.arctan(im/re).transpose().squeeze()

    return alpha, f, A, theta, ar_res


def orderProny(signal, order=2, epsilon=0, Fs=1):
    (alpha, f, A, theta, ar_res) = pronyDecomp(signal, order, epsilon, Fs)
    resid = np.sum(ar_res.bse**2)
    for currOrder in np.arange(order+2, 10, 2):
        (alphaCurr, fCurr, Acurr, thetaCurr, ar_resCurr) = pronyDecomp(signal, currOrder, epsilon, Fs)
        residCurr = np.sum(ar_resCurr.bse ** 2)
        if residCurr<resid:
            alpha = alphaCurr
            f = fCurr
            A = Acurr
            theta = thetaCurr
            resid = residCurr
            bse = ar_resCurr.bse
        else:  # Don't continue if residues are not decreasing.
            break
    return alpha, f, A, theta, ar_res


def chirpResample(signal, coeff=1, shift=None, dt=1):
    # coeff is slope degree of frequency line.
    # dt is 1 sample default. It assigns X axis sampling to determine coeff conveniently.
    X = np.arange(0, signal.size, dtype='int')
    signalResultFun = scipy.interpolate.interp1d(X, signal, kind='cubic')
    if shift is None:
        shift = np.arange(start=0, stop=signal.size, step=1)*dt*coeff+1
    dXnew = np.ones((1, signal.size)) / shift
    xNew = np.cumsum(dXnew)
    xNew = xNew[xNew < X.max()]
    return signalResultFun(xNew)


def trackResample(signal, track=1, t=None, phase_ini=None, aimed_phase=None, freqCoeff=1, targetedF=None):
    if t.size > 1:
        dt = t[1] - t[0]
    else:
        dt = t
    # Frequency track is assumed to be uniform default. It makes possible to change frequency freqCoeff times without frequency estimation.
    track = SM.check_like(track, signal)
    if phase_ini is None:
        phase_ini = np.cumsum(2*np.pi*track * dt)
    # Take the initial frequency default.
    if targetedF is None:
        targetedF = track[0]*freqCoeff
    dPhi = targetedF * 2 * np.pi * dt  # Phase step
    if aimed_phase is None:
        aimed_phase = np.arange(phase_ini[0], phase_ini[-1], dPhi)
    signalResultFun = scipy.interpolate.interp1d(phase_ini, signal, kind='cubic')
    equiPhased = signalResultFun(aimed_phase)
    equiPhasedTime = np.linspace(0, aimed_phase.size * dt, aimed_phase.size)
    return equiPhased, equiPhasedTime


def chirpProny(signal, order=2, epsilon=0, Fs=1, step=0, limit=0):
    # Take initial signal residues and parameters.
    dt = 1/Fs
    (alpha, f, A, theta, ar_res) = orderProny(signal, order, epsilon, Fs)
    order = f.size
    resid = np.sum(ar_res.bse**2)
    bse = ar_res.bse
    coefficient = 0
    # Try to compensate increasing frequency.
    for coeff in np.arange(step, limit, step):
        resampledChirp = chirpResample(signal, coeff, dt=dt)
        (alphaCurr, fCurr, Acurr, thetaCurr, ar_resCurr) = pronyDecomp(resampledChirp, order, epsilon, Fs)
        residCurr = np.std(ar_resCurr.bse)/np.std(resampledChirp)
        if residCurr<resid:
            alpha = alphaCurr
            f = fCurr
            A = Acurr
            theta = thetaCurr
            resid = residCurr
            coefficient = coeff
            bse = ar_resCurr.bse
        else:  # Don't continue if residues are not decreasing.
            break
    # Try to compensate decreasing frequency.
    for coeff in np.arange(-step, -limit, -step):
        resampledChirp = chirpResample(signal, coeff, dt=dt)
        (alphaCurr, fCurr, Acurr, thetaCurr, ar_resCurr) = pronyDecomp(resampledChirp, order, epsilon, Fs)
        residCurr = np.sum(ar_resCurr.bse ** 2)
        if residCurr<resid:
            alpha = alphaCurr
            f = fCurr
            A = Acurr
            theta = thetaCurr
            resid = residCurr
            coefficient = coeff
            bse = ar_resCurr.bse
        else:  # Don't continue if residues are not decreasing.
            break
    return alpha, f, A, theta, resid, coefficient


def rms(signal, omitnan=True):
    if omitnan:
        signal = signal[np.isnan(signal) == False]
    return np.sqrt(np.sum(signal**2)/signal.size)


def windMEXH(centrFreq, t, formFactor, carr=False):
    wind = 2 * np.pi * centrFreq * t
    psi = np.exp(-(wind ** 2) / 4 / formFactor)  # Form the window function
    if carr:
        psi *= np.cos(wind)
    return psi, wind


def wavSpectrum(psi, rect=0, level=0, fVectIni=None, Fs=1):
# rect = 0 -> complex window; rect > 0 and level > 0 -> drop low coefficients;
# rect == 2 -> real coefficients window to avoid phase distortions; rect == 3 -> rectangular real spectral window.
    if fVectIni is None:
        fVectIni = np.fft.rfftfreq(psi.size, 1 / Fs)
    windSpecSided = np.fft.rfft(psi)[0:len(fVectIni)]/psi.size
    windSpec = np.hstack((np.flip(windSpecSided[1:]), windSpecSided))
    windSpec = np.conj(windSpec)/np.max(np.abs(windSpec))
    freqz = np.hstack((-np.flip(fVectIni[1:]), fVectIni))
    supportIndexes = np.nonzero(np.abs(windSpec) >= level)
    supportFreqs = freqz[supportIndexes]
    if rect and level>0:
        zeroIndexes = np.array(np.nonzero(np.abs(windSpec) < level))
        if zeroIndexes.size:
            windSpec[zeroIndexes] = 0
        if rect == 3:
            windSpec[windSpec>0] = 1
    if rect == 2:
        windSpec = np.abs(windSpec) + 0j

    return windSpec, freqz, supportIndexes, supportFreqs


def mirrorExtendIndexes(signal, mirrorLen=0):
    num = np.arange(start=signal.size*mirrorLen+1, stop=0, step=-1, dtype='int')
    numEnd = np.arange(start=signal.size-2, stop=signal.size*(1-mirrorLen)-1, step=-1, dtype='int')
    return num, numEnd


def mirrorExtend(signal, mirrorLen=0):
    if mirrorLen:
        (num, numEnd) = mirrorExtendIndexes(signal, mirrorLen=mirrorLen)
        signal = np.hstack((signal[num], signal, signal[numEnd]))
    else:
        num=0
        numEnd=signal.size-2
    return signal, num, numEnd


def DFTbank(signal, Fs=1, df=None, fVectIni=None, rect=0, level=0.5, mirrorLen=0, freqLims=None, formFactor=16, **kwargs):
    (signal, num, numEnd) = mirrorExtend(signal, mirrorLen=mirrorLen)
    if signal.size % 2 == 1:
        signal = signal[1:]
    dt = 1/Fs
    t = np.arange(0, signal.size*dt, dt)
    # Get frequency vector for decomposition
    if not df:
        df = Fs/100  # 1/Fs
    if fVectIni is None:
        fVectIni = np.fft.rfftfreq(signal.size, 1 / Fs)
    fVect = np.unique(SM.closeInVect(fVectIni, np.arange(start=0, stop=Fs / 2, step=df))[0])
    if not freqLims is None:
        fVect = fVect[fVect<freqLims[-1]]
        if len(freqLims) == 2:
            fVect = fVect[fVect > freqLims[0]]
    spec = np.fft.rfft(signal)[0:len(fVectIni)] / signal.size  # Get one-sided FFT
    representation = np.zeros((fVect.size, signal.size))  # Columns are freqs, rows are time coefficients.
    if fVect[0] == 0:  # Consider zero frequency.
        start=1
        representation[0, :] += np.mean(signal)
    else:
        start=0
    for ai in range(start, fVect.size):
        # Obtain analysis window and it's width
        psi = windMEXH(fVect[ai], t, formFactor)[0]
        (windSpec, freqz) = wavSpectrum(psi, rect=rect, level=level, fVectIni=fVectIni, Fs=Fs)[0:2]
        # Shift the spectrum by needed frequencies and multiply by conjugated window spectrum.
        startIdx = np.nonzero(freqz == 0)[0][0] - np.nonzero(fVectIni == fVect[ai])[0][0]
        stopIdx = startIdx + fVectIni.size
        windSpec[np.arange(start=startIdx, stop=stopIdx, step=1, dtype='int')] *= spec  # Windowed spectrum.
        # Drop window function coefficients.
        windSpec[np.arange(start=0, stop=startIdx, step=1, dtype='int')] = 0
        windSpec[np.arange(start=stopIdx, stop=windSpec.size, step=1, dtype='int')] = 0
        oneSidedSpectrum = np.zeros_like(np.fft.rfftfreq(signal.size, 1 / Fs), dtype='complex')
        oneSidedSpectrum[0:(stopIdx-startIdx)] = windSpec[startIdx:stopIdx]
        representation[ai, :] = np.fft.irfft(oneSidedSpectrum)*t.size

    if mirrorLen:
        representation = representation[:, num[0]:numEnd[0]+num[0]+2]
    t = np.arange(0, representation.shape[1] * dt, dt)

    return representation, t, fVect


def scalogramFromRepresentation(representation, fVect=(-1,), firstDC=False):
    scalogram = np.zeros((representation.shape[0], 1), dtype='float64')
    for ai in range(scalogram.size):
        scalogram[ai] = np.std(representation[ai, :])
    if firstDC or fVect[0] == 0:
        scalogram[0] = representation[0, 0]
    return scalogram


def segmentedProny(signal, Fs=1, percentLength=10, percentOverlap=95, freqLengthHz=5, freqOverlapHz=0, order=2, lowFreq=90, highFreq=210):
    spec = np.fft.rfft(signal) / signal.size  # Get one-sided FFT
    fVectIni = np.fft.rfftfreq(signal.size, 1 / Fs)
    fVectCut = fVectIni[np.logical_and(fVectIni>lowFreq, fVectIni<highFreq)]
    (windowsFreq, freqWinIdxs) = windowing(fVectCut, freqLengthHz, (fVectIni[-1]-fVectIni[0])*freqOverlapHz/100)
    for bi in range(windowsFreq.shape[0]):
        specIndexes = np.logical_and(fVectIni>=windowsFreq[bi][0], fVectIni<=windowsFreq[bi][-1])
        currentSpec = np.zeros_like(spec)
        currentSpec[specIndexes] = spec[specIndexes]
        currentSignal = np.fft.irfft(currentSpec)*currentSpec.size
        (alphaCurr, fCurr, Acurr, thetaCurr, residCurr, coefficientCurr) = timeSegmentedProny(currentSignal, Fs=Fs, percentLength=percentLength, percentOverlap=percentOverlap)


def representProny(signal, representation, fVect, Fs=1, percentLength=10, percentOverlap=95, lowFreq=90, highFreq=210, level=0.5):
    f = []
    alpha = []
    A = []
    theta = []
    resid = []
    coefficient = []
    errors = []
    fVectCut = fVect[np.logical_and(fVect>=lowFreq, fVect<=highFreq)]
    dt = 1/Fs
    t = np.arange(0, signal.size*dt, dt)
    for bi in range(0, fVectCut.size):
        frInd = fVectCut[bi] == fVect
        currentSignal = representation[frInd, :][0]
        (alphaCurr, fCurr, Acurr, thetaCurr, residCurr, coefficientCurr) = timeSegmentedProny(currentSignal, Fs=Fs, percentLength=percentLength, percentOverlap=percentOverlap)
        psi = windMEXH(fVect[frInd], t, 1024)[0]
        supportFreqs = wavSpectrum(psi, rect=False, level=level, fVectIni=fVect, Fs=Fs)[3]
        supportFreqs += fVect[frInd]
        errorsCurr = []
        for ci in range(0, len(fCurr)):
            fTemp = fCurr[ci][fCurr[ci] > 0]
            errorsTemp = np.zeros_like(fTemp)
            for ai in range(0, fTemp.size):
                errorsTemp[ai] = np.max((fTemp[ai]-supportFreqs[-1], supportFreqs[0]-fTemp[ai]))  # Outlying of the current frequency out of filter range.
            errorsTemp[errorsTemp < 0] = 0
            errorsCurr.append(errorsTemp)
        alpha.append(alphaCurr)
        f.append(fCurr)
        A.append(Acurr)
        theta.append(thetaCurr)
        resid.append(residCurr)
        coefficient.append(coefficientCurr)
        errors.append(errorsCurr)
    return alpha, f, A, theta, resid, coefficient, errors


def timeSegmentedProny(signal, Fs=1, percentLength=10, samplesLenWin=None, percentOverlap=95, order=2, useChirp=False):
    if samplesLenWin is None:
        samplesLenWin = np.round(signal.size * percentLength / 100)
    (windows, timeWinIdxs) = windowing(signal, int(samplesLenWin), percentOverlap)
    f = []  # np.zeros((0, order))
    alpha = []
    A = []
    theta = []
    resid = np.zeros((0, 1))
    coefficient = np.zeros((0, 1))
    for ai in range(windows.shape[0]):
        if useChirp:
            (alphaCurr, fCurr, Acurr, thetaCurr, residCurr, coefficientCurr) = chirpProny(windows[ai, :], Fs=Fs, order=order, step=0.05, limit=0.9)
        else:
            (alphaCurr, fCurr, Acurr, thetaCurr, ar_res) = pronyDecomp(windows[ai, :], order, Fs=Fs)
            residCurr = np.std(ar_res.bse)/np.std(windows[ai, :])
            coefficientCurr = 0
        alpha.append(alphaCurr)
        f.append(fCurr)
        A.append(Acurr)
        theta.append(thetaCurr)
        resid = np.vstack((resid, residCurr))
        coefficient = np.vstack((coefficient, coefficientCurr))

    return alpha, f, A, theta, resid, coefficient


def thresholdRepreProny(representation, fVect, **kwargs):

    Fs = kwargs.get('Fs', 1)
    percentLength = kwargs.get('percentLength', 5)
    percentOverlap = kwargs.get('percentOverlap', 50)
    periodsNum = kwargs.get('periodsNum', None)
    secondsNum = kwargs.get('secondsNum', None)
    powerNorm = kwargs.get('powerNorm', 0)
    (lowFreq, highFreq) = kwargs.get('roughFreqs', (kwargs.get('lowFreq', 70), kwargs.get('highFreq', 150)))
    #lowFreq = kwargs.get('lowFreq', 90)
    #highFreq = kwargs.get('highFreq', 210)
    hold = kwargs.get('hold', 1)
    useChirp = kwargs.get('useChirp', False)
    order = kwargs.get('order', 2)

    dt = 1/Fs
    t = np.arange(start=0, stop=representation.shape[-1]*dt, step=dt)
    f = np.zeros_like(t)
    alpha = np.zeros_like(t)
    A = np.zeros_like(t)
    theta = np.zeros_like(t)
    resid = np.ones_like(t)*np.inf
    coefficient = np.zeros_like(t)
    (lowFreq, highFreq) = kwargs.get('roughFreqs', (lowFreq, highFreq))
    fVectCut = kwargs.get('fVectCut', fVect[np.logical_and(fVect>=lowFreq, fVect<=highFreq)])
    # Window length is assigned in percents of the whole signal.
    samplesLenWin = kwargs.get('samplesLenWin', np.round(t.size*percentLength/100))
    if secondsNum:
        samplesLenWin = np.round(secondsNum/dt)

    """""
    ???
    fRough = kwargs.get('fRough', [])
    if len(fRough) > 0:
        # Limit frequencies for analysing.
        lims = kwargs.get('roughFreqs', fVect)
        fRough[fRough<lims[0]] = lims[0]
        if len(lims) > 1:
            fRough[fRough > lims[0]] = lims[-1]
        # Check track frequencies are in the representation frequency vector.
        if not any(np.isinf(fVect)):
            fRough = SM.closeInVect(fVect, fRough)
    """""

    """""
    fig = plt.figure(figsize=(8, 6))
    grid = matplotlib.gridspec.GridSpec(1, 1)
    ax_F = fig.add_subplot(grid[0])
    ax_F.grid()
    ax_Err = ax_F.twinx()
    ax_Err.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    fig.show()
    cmap1 = plt.cm.get_cmap('magma', fVectCut.size)
    """""

    for ai in range(0, fVectCut.size):
        (fCur, frInd) = SM.closeInVect(fVect, fVectCut[ai])
        if periodsNum:
            samplesLenWin = np.round(1/fVectCut[ai]/dt*periodsNum)
        currHold = rms(representation[frInd, :])*hold
        timeIndexes = np.nonzero(representation[frInd, :]>=currHold)
        if  timeIndexes[1].size<10:
            continue
        timeIndexes = np.arange(start=timeIndexes[1][0], stop=timeIndexes[1][-1], step=1, dtype='int')
        signal = representation[frInd, timeIndexes].transpose()
        if powerNorm:
            signal *= powerNorm/(np.sum(signal**2)/signal.size)
        # Get time indexes windows to situate obtained values.
        tWinIndexes = windowing(timeIndexes, int(samplesLenWin), percentOverlap, dtype='int')[0]
        # Take only indexes crossed with unestimated regions if the current frequency is not included in track.???
        (alphaCurr, fCurr, Acurr, thetaCurr, residCurr, coefficientCurr) = timeSegmentedProny(signal,
              Fs=Fs, samplesLenWin=samplesLenWin, percentOverlap=percentOverlap, order=order, useChirp=useChirp)

        """""
        win1 = []
        for di in range(len(tWinIndexes)):
            win1.append(tWinIndexes[di][0])

        tw = t[np.array(win1)]
        ax_F.plot(tw, fCurr[:, 1], color=cmap1(ai))
        ax_Err.plot(tw, residCurr.transpose(), linestyle=':', color=cmap1(ai), label='f0 = {}'.format(fVectCut[ai]))
        ax_Err.legend()
        """""

        fCurr = np.array(fCurr)
        alphaCurr = np.array(alphaCurr)
        Acurr = np.array(Acurr)
        thetaCurr = np.array(thetaCurr)
        residCurr = residCurr.transpose()
        coefficientCurr = coefficientCurr.transpose()
        for bi in range(tWinIndexes.shape[0]):
            currIdxs = tWinIndexes[bi, :]
            appropriateIdxs = currIdxs[resid[currIdxs]>residCurr[0, bi]]
            if kwargs.get('centralFreq', False):
                fCurr[bi, 1] = fVectCut[ai]
            f[appropriateIdxs] = fCurr[bi, 1]
            alpha[appropriateIdxs] = alphaCurr[bi, 1]
            A[appropriateIdxs] = Acurr[bi, 1]
            theta[appropriateIdxs] = thetaCurr[bi, 1]
            resid[appropriateIdxs] = residCurr[0, bi]
            coefficient[appropriateIdxs] = coefficientCurr[0, bi]
        pass

    result = alpha, f, A, theta, resid, coefficient
    errorThresh = kwargs.get('errorThresh', np.inf)
    dummyVal = kwargs.get('dummyVal')
    diffHold = kwargs.get('diffHold')
    for ai in range(len(result)):
        holdErr = errorThresh*np.median(resid)
        dropIndexes = np.hstack(np.nonzero(resid >= holdErr))
        if not dummyVal is None:
            result[ai][dropIndexes] = dummyVal
            if not diffHold is None:
                diffHoldTemp = diffHold*np.mean(np.diff(resid[resid<holdErr]))
                dropIndexes = diffThreshold(result[ai], diffHoldTemp, dummyValue=result[ai][0])
                result[ai][dropIndexes] = dummyVal

    return result

def fftFiltrate(signal, freqLims, Fs=1):
    fVect = np.fft.rfftfreq(signal.size, 1 / Fs)
    oneSidedSpectrum = np.fft.rfft(signal)[0:len(fVect)] / signal.size
    oneSidedSpectrum[fVect<freqLims[0]] = 0
    oneSidedSpectrum[fVect>freqLims[1]] = 0
    return fVect, np.fft.irfft(oneSidedSpectrum) * signal.size

def conventionalProny(signal, **kwargs):
    Fs = kwargs.get('Fs', 1)
    (lowFreq, highFreq) = kwargs.get('roughFreqs', (kwargs.get('lowFreq', 70), kwargs.get('highFreq', 150)))
    signalFiltrated = np.zeros((1, signal.size), dtype=signal.dtype)
    fVect, signalFiltrated[0, :] = fftFiltrate(signal, (lowFreq, highFreq), Fs=Fs)
    centralFreq = np.zeros((1, 1), dtype='float64')
    centralFreq[0, 0] = np.mean(np.array((lowFreq, highFreq)))
    (alpha, f, A, theta, resid, coefficient) = thresholdRepreProny(signalFiltrated, centralFreq, dummyVal=np.nan, **kwargs)
    return alpha, f, A, theta, resid, coefficient, signalFiltrated, centralFreq

def pronyParamsEst(signal, **kwargs):
    Fs = kwargs.get('Fs', 1)
    kwargs.update({'hold': kwargs.get('hold', 1.4)})
    formFactor = kwargs.get('formFactor', 1024)
    formFactorCurr = formFactor[0] if type(formFactor) in [tuple, list] else formFactor
    secondsNum = kwargs.get('secondsNum')
    if type(secondsNum) in [tuple, list]:
        kwargs.update({'secondsNum': secondsNum[0]})
        secondsNum = secondsNum[-1]
    periodsNum = kwargs.get('periodsNum')
    if type(periodsNum) in [tuple, list]:
        kwargs.update({'periodsNum': periodsNum[0]})
        periodsNum = periodsNum[-1]
    iterations = kwargs.get('iterations', 0)
    (lowFreq, highFreq) = kwargs.get('roughFreqs', (kwargs.get('lowFreq', 70), kwargs.get('highFreq', 150)))
    if kwargs.get('roughFreqs', 0) and iterations:  # Get preliminary track to estimate frequency borders.
        # ///Pitch detection and rough estimation///
        (representation, t0, fVectNew) = DFTbank(signal, rect=2, level=0.2, Fs=Fs, mirrorLen=0.15, df=kwargs.get('df', 1),
                                                    freqLims=(lowFreq, highFreq), formFactor=formFactorCurr)
        (alpha, f, A, theta, resid, coefficient) = thresholdRepreProny(representation, fVectNew, dummyVal=np.nan, **kwargs)
        # Get cumulative occurrence frequency of each value - number of values less each threshold.
        cumF = distributLaw(f, fVectNew)[0]
        # Define interested pitch band as increasing occurrence rate distance.
        idxMin = int(np.nonzero(cumF > np.max(cumF) * 0.1)[0][0])
        idxMax = min(int(np.nonzero(cumF < np.max(cumF) * 0.98)[0][-1]), fVectNew.size-1)
        if kwargs.get('plotGraphs', 0) == 2:
            SM.plotUnder(fVectNew, ((cumF, cumF[idxMin], cumF[idxMax]),))
            print('Estimated pitch limits:')
            print((fVectNew[idxMin], fVectNew[idxMax]))
        kwargs.update({'roughFreqs': (fVectNew[idxMin], fVectNew[idxMax]), 'formFactor': formFactor[1:]})
        kwargs.update({'iterations': iterations-1})  # Subtract recursive iterations counter.
        kwargs.update({'secondsNum': secondsNum})
        kwargs.update({'periodsNum': periodsNum})
        (alpha, f, A, theta, res, coefficient, representation1, fVectNew1) = pronyParamsEst(signal, **kwargs)
        return alpha, f, A, theta, res, coefficient, representation, fVectNew
    else:
        # ///Pitch estimation///
        (representation, t0, fVectNew) = DFTbank(signal, rect=2, level=0.2, Fs=Fs, mirrorLen=0.15, df=kwargs.get('df', 1),
                                                 freqLims=(lowFreq, highFreq), formFactor=formFactorCurr)
        (alpha, f, A, theta, resid, coefficient) = thresholdRepreProny(representation, fVectNew, dummyVal=np.nan, **kwargs)
    if kwargs.get('t') and kwargs.get('plotGraphs', 0) == 2:
        t = SM.makeNP(kwargs.get('t', 1))
        t = SM.genTime(maxTime=t, Fs=Fs) if t.size == 1 else t
        fInd = int(SM.closeInVect(fVectNew, np.nanmedian(f))[1])
        print('Frequency RMSE:')
        print(rms(f-kwargs.get('freq'), True))
        SM.plotUnder(t, (signal, alpha,  (f, kwargs.get('freq'))), secondParam=resid,
                  labels=('Signal', 'Estimated decay', ('Estimated frequency', 'Real frequency')), secondLabel='Error',
                  secondLim=(0, np.nanmedian(resid)*1.1), xlabel='Time, sec', ylabel=('Modeled signal', 'Exponential decay', 'Frequency'), secLabel='Approximation error',
                  yLims=(None, (np.nanmedian(alpha)-np.nanstd(alpha)*3, np.nanmedian(alpha)+np.nanstd(alpha)*3), None))
    return alpha, f, A, theta, resid, coefficient, representation, fVectNew


def trackEstimation(representation, fRough, fVect=(-np.inf, np.inf), **kwargs):
    # Limit frequencies for analysing.
    lims = kwargs.get('roughFreqs', fVect)
    fRough[fRough<lims[0]] = lims[0]
    if len(lims) > 1:
        fRough[fRough > lims[0]] = lims[-1]
    # Check track frequencies are in the representation frequency vector.
    if not any(np.isinf(fVect)):
        fRough = SM.closeInVect(fVect, fRough)


def distributLaw(values, scale=None, dummyVal=None):
    f = np.hstack(values) if type(values) in [tuple, list] else np.zeros_like(values) + values  # Copy and delete NaNs
    f = f[np.isnan(f) == False]
    if scale is None:
        step = np.mean(np.diff(np.sort(f)))
        if np.isnan(step):
            scale = np.array([0.9, 1, 1.1])*f[0]
        else:
            scale = np.arange(np.min(f), np.max(f), step)
    cumF = np.zeros_like(scale)
    for ci in range(scale.size):
        cumF[ci] = np.count_nonzero(f < scale[ci])
    if not dummyVal is None:
        return cumF, scale, np.diff(np.hstack((dummyVal, cumF)))  # Compute probability density.
    return cumF, scale


def scalogramFinding(representation=None, fVect=None, **kwargs):
    if representation is None:
        #signal = kwargs.get('signal')
        (representation, t, fVect) = DFTbank(**kwargs)
    repFreq = representationTrack(representation, fVect)
    sc = scalogramFromRepresentation(representation)
    sc = (sc/np.max(sc)).squeeze()
    peaks = scipy.signal.find_peaks(sc)
    if kwargs.get('plotGraphs', 0) == 2:
        fig=plt.figure()
        plt.plot(fVect, sc)
        plt.plot(fVect[peaks[0]], sc[peaks[0]], "x")
        plt.xlabel('Frequency, Hz')
        plt.ylabel('Normalized scalogram')
    return repFreq, sc, peaks



def diffThreshold(signal, hold, dummyValue=None):
    if dummyValue is not None:
        dff = np.diff(np.hstack((dummyValue, signal)))
    else:
        dff = np.diff(signal)
    indexes = np.abs(dff) < hold
    return indexes, signal[indexes]


def representationTrack(representation, fVect):
    instFreq = np.zeros_like(representation[0, :])
    representationEnv = np.zeros_like(representation)
    # Get envelope along each frequency to define the most prominent time-frequency regions.
    for ai in range(representation[:, 0].size):
        representationEnv[ai, :] = np.abs(hilbert(representation[ai, :]))
    for ai in range(instFreq.size):
        freqInd = np.argmax(representationEnv[:, ai])
        instFreq[ai] = fVect[freqInd]
    return instFreq


def hilbertTrack(representation, fVect, Fs, f0):
    index = SM.closeInVect(fVect, f0)[1]
    signal = hilbert(representation[index, :])
    phase = np.unwrap(np.angle(signal))
    frequency = np.hstack((np.nan, (np.diff(phase) / (2.0 * np.pi) * Fs).squeeze()))
    frequency[frequency.size-50:] = np.nan
    frequency[0:50] = np.nan
    return frequency


def validateTrack(**kwargs):
    # Get sequences with frequency diffs less assigned hold and having enough width.
    f = kwargs.get('track')
    f[np.isnan(f)] = np.nanmax(f)
    dff = np.abs(np.diff(np.hstack((f[0], f))))  # Frequency jumps.
    # Indexes of significant frequency jumps.
    dff[dff < kwargs.get('hold', 0)] = 0  # Nullify low difference hops.
    if np.nonzero(dff)[0].size == 0:
        return (np.arange(f.size),)
    # Check length and number of continuous parts of track
    contLen = np.round(SM.makeNP(kwargs.get('contLen', 40))*f.size/100)
    maxPlateau = None  # Assign the greater plateau limits as a previous one.
    validatedRanges = []
    for ai in range(contLen.size):
        peaks = scipy.signal.find_peaks(-np.hstack((np.max(dff), dff, np.max(dff))), plateau_size=(contLen[ai], maxPlateau))
        lb = peaks[1].get('left_edges')-1
        rb = peaks[1].get('right_edges')-1
        if lb.size > ai:  # Required number of pieces of assigned length.
            for bi in range(lb.size):
                indexes = np.arange(lb[bi], rb[bi])
                truth = kwargs.get('truthVals')
                if truth is None or np.mean(truth[indexes]-f[indexes]) < kwargs.get('hold', 0):
                    validatedRanges.append(indexes)
        maxPlateau = contLen[ai]
    return validatedRanges

def medianSmooth(signal, samplesWidth=None, t=None, secondsWidth=None):
    if t is not None and secondsWidth is not None:
        if t.size > 1:
            dt = t[1] - t[0]
        else:
            dt = t
        samplesWidth = secondsWidth/dt
    signalWin, samples = windowing(signal, fftsize=samplesWidth, hop=1)
    signalNew = np.zeros_like(signal)+np.nan
    for ai in range(signalWin.shape[0]):
        currentWindow = signalWin[ai, :]
        signalNew[int(ai+np.floor(currentWindow.size/2))] = np.median(currentWindow)  # Estimate sample of the smoothed signal at window center.
    return signalNew

def getList(param):
    if not type(param) in [tuple, list, type(None)]:
        param = [param]
    return param

def setList(param, name, kwargs):
    if not type(param) in [tuple, list]:
        kwargs.update({name: param})
    else:
        kwargs.update({name: param[1:]})
    return kwargs

def randomDeviation(argDict, **kwargs):
    paramNames = getList(kwargs.get('paramNames'))
    percentDeviation = SM.check_like(kwargs.get('percentDeviation'), np.arange(len(paramNames)))
    absoleteDeviation = SM.check_like(kwargs.get('absoleteDeviation'), percentDeviation)
    negativeDeviation = SM.check_like(kwargs.get('negativeDeviation', 0), percentDeviation)
    for ai, name in enumerate(paramNames):
        if not type('name') is str:
            raise(TypeError)
        param = SM.makeNP(argDict.get(name))
        if not type(percentDeviation) is None:
            percentVariation = SM.makeNP(percentDeviation[ai])
            absoleteVariation = percentVariation*param/100
        elif not type(absoleteDeviation) is None:
            absoleteValue = SM.makeNP(absoleteDeviation[ai])
            if not np.isnan(absoleteValue):
                absoleteVariation = absoleteValue
        else:
            raise (TypeError)
        if np.any(np.isnan(absoleteVariation)):
            raise(ValueError)
        normalizedRandom = SM.makeNP([random.random() for i in range(param.size)])
        if negativeDeviation[ai]:
            deviation = absoleteVariation*(normalizedRandom*2-1)
        else:
            deviation = absoleteVariation*normalizedRandom
        argDict.update({name: param+deviation})

def getSamplesNum(kwargs):
    periodsNum = kwargs.get('periodsNum', None)
    secondsNum = kwargs.get('secondsNum', None)
    samplesLenWin = kwargs.get('samplesLenWin', None)
    percentLength = kwargs.get('percentLength', None)

import numpy as np
from statsmodels.tsa.ar_model import AutoReg as AR
import scipy
import matplotlib
from matplotlib import pyplot as plt
from scipy.signal import hilbert
import signalModeling as SM


def awgn(signal, SNRdB=None, SNRlin=None, sigPower=None):
    if not SNRlin:
        SNRlin = 0
    if not SNRdB is None:
        SNRlin = SNRlin + 10 ** (SNRdB / 10)
    if not sigPower:
        sigPower = np.sum(signal**2)/signal.size
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


def windowing(x, fftsize=1024, overlap=4, returnShortSignal=True, joinLastSample=False, dtype=None):
    hop = int(fftsize * (100 - overlap)/100)
    hop = max(hop, 1)
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
    A = np.abs(h).transpose()
    theta = np.arctan(im/re).transpose()

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
    if not shift:
        shift = np.arange(start=0, stop=signal.size, step=1)*dt*coeff+1
    dXnew = np.ones((1, signal.size)) / shift
    xNew = np.cumsum(dXnew)
    xNew = xNew[xNew < X.max()]
    return signalResultFun(xNew)


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


def rms(signal):
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
    windSpecSided = np.fft.rfft(psi)/psi.size
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


def DFTbank(signal, Fs=1, df=None, fVectIni=None, rect=0, level=0.5, mirrorLen=0, freqLims=None, formFactor=16):
    (signal, num, numEnd) = mirrorExtend(signal, mirrorLen=mirrorLen)
    dt = 1/Fs
    t = np.arange(0, signal.size*dt, dt)
    # Get frequency vector for decomposition
    if not df:
        df = Fs/100  # 1/Fs
    if not fVectIni:
        fVectIni = np.fft.rfftfreq(signal.size, 1 / Fs)
    fVect = np.unique(SM.closeInVect(fVectIni, np.arange(start=0, stop=Fs / 2, step=df))[0])
    if not freqLims is None:
        fVect = fVect[fVect<freqLims[-1]]
        if len(freqLims) == 2:
            fVect = fVect[fVect > freqLims[0]]
    spec = np.fft.rfft(signal) / signal.size  # Get one-sided FFT
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
        representation[ai, :] = np.fft.irfft(windSpec[np.arange(start=startIdx, stop=stopIdx, step=1, dtype='int')])*t.size

    if mirrorLen:
        representation = representation[:, num[0]:numEnd[0]+num[0]+2]

    return representation, t, fVect


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
            residCurr = np.sum(ar_res.bse ** 2)
            coefficientCurr = 0
        alpha.append(alphaCurr)
        f.append(fCurr)
        A.append(Acurr)
        theta.append(thetaCurr)
        resid = np.vstack((resid, residCurr))
        coefficient = np.vstack((coefficient, coefficientCurr))

    return alpha, f, A, theta, resid, coefficient


def thresholdRepreProny(representation, fVect, Fs=1, percentLength=5, percentOverlap=50, periodsNum=None, mirrorLen=0, errorThresh=None, lowFreq=90, highFreq=210, hold=1, useChirp=False, order=2):
    dt = 1/Fs
    t = np.arange(start=0, stop=representation.shape[1]*dt, step=dt)
    f = np.zeros_like(t)
    alpha = np.zeros_like(t)
    A = np.zeros_like(t)
    theta = np.zeros_like(t)
    resid = np.ones_like(t)*np.inf
    coefficient = np.zeros_like(t)
    fVectCut = fVect[np.logical_and(fVect>=lowFreq, fVect<=highFreq)]
    # Window length is assigned in percents of the whole signal.
    samplesLenWin = np.round(t.size*percentLength/100)

    for ai in range(0, fVectCut.size):
        frInd = fVectCut[ai] == fVect
        if periodsNum:
            samplesLenWin = np.round(1/fVectCut[ai]/dt*periodsNum)
        currHold = rms(representation[frInd, :])*hold
        timeIndexes = np.nonzero(representation[frInd, :]>=currHold)
        if  timeIndexes[1].size<10:
            continue
        timeIndexes = np.arange(start=timeIndexes[1][0], stop=timeIndexes[1][-1], step=1, dtype='int')
        signal = representation[frInd, timeIndexes].transpose()
        # Get time indexes windows to situate obtained values.
        tWinIndexes = windowing(timeIndexes, int(samplesLenWin), percentOverlap, dtype='int')[0]
        (alphaCurr, fCurr, Acurr, thetaCurr, residCurr, coefficientCurr) = timeSegmentedProny(signal,
              Fs=Fs, samplesLenWin=samplesLenWin, percentOverlap=percentOverlap, order=order, useChirp=useChirp)
        '''''
        if useChirp:
            (alphaCurr, fCurr, Acurr, thetaCurr, residCurr, coefficientCurr) = chirpProny(signal, Fs=Fs, order=order, step=0.05, limit=0.9)
        else:
            (alphaCurr, fCurr, Acurr, thetaCurr, ar_res) = pronyDecomp(signal, order, Fs=Fs)
            residCurr = np.sum(ar_res.bse ** 2)
            coefficientCurr = 0
        '''''
        fCurr = np.array(fCurr)
        alphaCurr = np.array(alphaCurr)
        residCurr = residCurr.transpose()
        coefficientCurr = coefficientCurr.transpose()
        for bi in range(tWinIndexes.shape[0]):
            currIdxs = tWinIndexes[bi, :]
            appropriateIdxs = currIdxs[resid[currIdxs]>residCurr[0, bi]]
            f[appropriateIdxs] = fCurr[bi, 1]
            alpha[appropriateIdxs] = alphaCurr[bi, 1]
            #A[appropriateIdxs] = Acurr[1]
            #theta[appropriateIdxs] = thetaCurr[1]
            resid[appropriateIdxs] = residCurr[0, bi]
            coefficient[appropriateIdxs] = coefficientCurr[0, bi]

    if errorThresh:
        holdErr = errorThresh*np.min(resid)
        dropIndexes = np.hstack(np.nonzero(resid>holdErr))
        resid[dropIndexes] = np.inf
        f[dropIndexes] = 0
        alpha[dropIndexes] = 0

    return alpha, f, A, theta, resid, coefficient


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


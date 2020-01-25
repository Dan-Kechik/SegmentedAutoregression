import numpy as np
from statsmodels.tsa.ar_model import AR
import scipy


def awgn(signal, SNRdB=None, SNRlin=None, sigPower=None):
    if not SNRlin:
        SNRlin = 0
    if SNRdB:
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
        noise = np.fft.irfft(noiseSpec)
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


def windowing(x, fftsize=1024, overlap=4):
    hop = int(fftsize * (100 - overlap)/100)
    signal = np.array([x[i:i+fftsize] for i in range(0, len(x)-fftsize, hop)])
    samples = np.array([i for i in range(0, len(x)-fftsize, hop)])
    return signal, samples


def pronyDecomp(signal, order, epsilon=0, Fs=1):
    dt = 1/Fs
    ar_mod = AR(signal)
    ar_res = ar_mod.fit(order)
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

    return alpha, f, A, theta, np.sum(ar_res.resid**2)


def chirpResample(signal, coeff=1, shift=None, dt=1):
    # coeff is slope degree of frequency line.
    # dt is 1 sample default. It assigns X axis sampling to determine coeff conveniently.
    X = np.arange(0, signal.size, dtype='int')
    signalResultFun = scipy.interpolate.interp1d(X, signal, kind='cubic')
    if not shift:
        shift = np.arange(start=0, stop=signal.size, step=1)*(coeff*dt)+1
    dXnew = np.ones((1, signal.size)) / shift
    xNew = np.cumsum(dXnew)
    xNew = xNew[xNew < X.max()]
    return signalResultFun(xNew)


def chirpProny(signal, order, epsilon=0, Fs=1, step=0, limit=0):
    # Take initial signal residues and parameters.
    dt = 1/Fs
    (alpha, f, A, theta, resid) = pronyDecomp(signal, order, epsilon, Fs)
    predictErr = np.sum(resid)
    coefficient = 0
    # Try to compensate increasing frequency.
    for coeff in np.arange(step, limit, step):
        resampledChirp = chirpResample(signal, coeff, dt=dt)
        (alphaCurr, fCurr, Acurr, thetaCurr, residCurr) = pronyDecomp(resampledChirp, order, epsilon, Fs)
        if residCurr<resid:
            alpha = alphaCurr
            f = fCurr
            A = Acurr
            theta = thetaCurr
            resid = residCurr
            coefficient = coeff
        else:  # Don't continue if residues are not decreasing.
            break
    # Try to compensate decreasing frequency.
    for coeff in np.arange(-step, -limit, -step):
        resampledChirp = chirpResample(signal, coeff, dt=dt)
        (alphaCurr, fCurr, Acurr, thetaCurr, residCurr) = pronyDecomp(resampledChirp, order, epsilon)
        if residCurr<resid:
            alpha = alphaCurr
            f = fCurr
            A = Acurr
            theta = thetaCurr
            resid = residCurr
            coefficient = coeff
        else:
            break
    return alpha, f, A, theta, resid, coefficient
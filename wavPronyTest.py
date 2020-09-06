import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from scipy.signal import hilbert, chirp
from scipy.io import wavfile
import pandas
import scipy
import pronyAnalysis as pa
import signalModeling as sm
import os
from matplotlib import cm
import copy
import dill
from multiprocessing import Pool
import datetime as dtm

RootFold = r'D:\Repository\ComputeFramework\Classifier\rawdataIni'  # r'D:\Repository\Vibration Records\Стенд ВБХ скоростные режимы\25_10\393.1'  # r'D:\Repository\ComputeFramework\Classifier\rawdata'  # r'D:\Repository\ComputeFramework\FunctionalTesting\ftpTemp\bearing6213NormOr_constSpeed\rawdata'  #
downsampl = 50
upCoefficient = 10  # Divide Fs to translate spectrum
plotGraphs = 1
targetedFrequency = 10
roughFreq = (20, 250)
processesNumber=6

def main():
    fList = os.listdir(RootFold)
    wavList = []
    processedWavs = []
    for ai in fList:
        if ai.endswith('.wav'):
            wavList.append(ai)
    # wavList=['22.wav']
    if processesNumber:
        with Pool(processesNumber) as pool:
            for r in pool.imap_unordered(processSingleFile, wavList):
                processedWavs.append(r)
    else:
        for currentFile in wavList:
            processSingleFile(currentFile)

def processSingleFile(currentFile):
    myFile = os.path.join(RootFold, currentFile)
    if os.path.exists(myFile):
        print('========='+currentFile+'=========\n\n')
        Fs, signal = wavfile.read(myFile)
        Fs2 = round(Fs/downsampl)
        t = sm.genTime(signal=signal, Fs=Fs)
        (resampled_x, resampled_t) = scipy.signal.resample(signal, round(len(signal) / downsampl), t=t)

        fileName = currentFile.rstrip('.wav')
        csvName = os.path.join(RootFold, fileName+'.csv')
        if os.path.exists(os.path.join(RootFold, csvName)):
            realFreq = pandas.read_csv(csvName).to_numpy(dtype='float64')
            csvTime = realFreq[:, 0]
            realFreq = realFreq[:, 1] / 2
            tachoTime = [dtm.datetime.fromtimestamp(i) for i in csvTime]
            tachoTime = [i - tachoTime[0] for i in tachoTime]
            tachoTime = np.array([i.total_seconds() for i in tachoTime])

        kwargs = {'Fs': Fs2*upCoefficient, 'plotGraphs': plotGraphs}
        kwargs.update({'roughFreqs': roughFreq, 'iterations': 1, 'formFactor': (32, 64), 'hold': 0})
        (alpha, f, A, theta, res, coefficient, representation, fVectNew) = pa.pronyParamsEst(resampled_x, secondsNum=0.005, percentOverlap=75, **kwargs)  #secondsNum=0.005
        print('Estimated track')
        resampled_x = resampled_x[0:f.size]
        resampled_t = resampled_t[0:f.size]
        if plotGraphs:
            fig = sm.plotRepresentation(resampled_t, representation, fVectNew, roughFreq)
            if os.path.exists(csvName):
                fig.axes[0].plot(tachoTime, realFreq*upCoefficient, color='r', label='Tacho track')
            fig.axes[0].plot(resampled_t[0:f.size], f[0:resampled_t.size], label='Estimated track')
            print('Plotted estimated track')

        fSmooth = pa.medianSmooth(f, t=resampled_t, secondsWidth=0.5)
        resampled_x = resampled_x[~np.isnan(fSmooth)]
        resampled_t = resampled_t[~np.isnan(fSmooth)]
        fSmooth = fSmooth[~np.isnan(fSmooth)]
        print('Smoothed track')
        if plotGraphs:
            fig.axes[0].plot(resampled_t, fSmooth, label='Smoothed track')
            fig.axes[0].set_xlabel('Time, sec')
            fig.axes[0].set_ylabel('Frequency, Hz')
            fig.axes[0].set_title("Initial signal and it's track")
            fig.axes[0].legend()
            print('Plotted smoothed track')
        '''
        with open('buff', "rb") as f:
            kws = dill.load(f)
        resampled_x = kws['resampled_x']
        resampled_t = kws['resampled_t']
        fSmooth = kws['fSmooth']
        '''''
        validIdxs, idx = validateTrack(fSmooth, secondsLen=5, t=resampled_t)
        indexes = validIdxs[idx]
        indexes = np.arange(indexes[0], indexes[1])
        equiPhased, equiPhasedTime = pa.trackResample(resampled_x[indexes], track=fSmooth[indexes]/upCoefficient, t=resampled_t[indexes], targetedF=targetedFrequency)
        print('Resampled signal')
        if plotGraphs:
            #(alphaR, fR, AR, thetaR, resR, coefficientR, representationR, fVectNewR) = pa.pronyParamsEst(equiPhased, secondsNum=0.05, percentOverlap=75, **kwargs)  #secondsNum=0.005 periodsNum=10
            # equiPhasedTime = equiPhasedTime[0:fR.size]
            (representationR, t0, fVectNewR) = pa.DFTbank(equiPhased, rect=2, level=0.2, Fs=Fs2 * 10, df=0.5, freqLims=(0, 250), formFactor=128)
            figR = sm.plotRepresentation(equiPhasedTime, representationR, fVectNewR, roughFreq)
            # figR.axes[0].plot(equiPhasedTime, fR, label='Estimated track')
            figR.axes[0].set_xlabel('Time, sec')
            figR.axes[0].set_ylabel('Frequency, Hz')
            figR.axes[0].set_title("Resampled signal and it's track")
            figR.axes[0].legend()
            print('Plotted resampled signal')

            figSpec = plt.figure()
            grid = matplotlib.gridspec.GridSpec(1, 1)
            ax_specWav = figSpec.add_subplot(grid[0])
            fVectIni = np.fft.rfftfreq(signal.size, 1 / Fs)
            specIni = np.fft.rfft(signal) / signal.size
            ax_specWav.plot(fVectIni, abs(specIni), label='Initial')
            fVectRes = np.fft.rfftfreq(equiPhased.size, 1 / Fs2)
            specRes = np.fft.rfft(equiPhased) / equiPhased.size
            ax_specWav.plot(fVectRes, abs(specRes), label='Resampled')
            ax_specWav.set_xlim(0, 50)
            figSpec.axes[0].set_xlabel('Frequency, Hz')
            figSpec.axes[0].set_ylabel('Amplitude')
            figSpec.axes[0].set_title('Comparison of spectrums')
            figSpec.axes[0].legend()
            print('Plotted spectrums')
            '''''
            with open(os.path.join('Out', 'resampledData', fileName+'.pkl'), "wb") as fl:
                kwSave = {'fig': fig, 'figR': figR, 'figSpec': figSpec}
                dill.dump(kwSave, fl)
                fl.close()
            '''

            fig.savefig(os.path.join('Out', 'resampledData', fileName+'_spectrogram.png'), bbox_inches='tight')
            figR.savefig(os.path.join('Out', 'resampledData', fileName+'_spectrogram_resampled.png'), bbox_inches='tight')
            figSpec.savefig(os.path.join('Out', 'resampledData', fileName+'_spectrum.png'), bbox_inches='tight')
            print('Saved plotted graphs')

            plt.close('all')

        equiPhased = equiPhased/np.max(np.abs(equiPhased))
        myFile = os.path.join('Out', 'resampledData', currentFile)
        scipy.io.wavfile.write(myFile, Fs2, equiPhased)

        return myFile
    else:
        return ''

def validateTrack(f, percentLenValid=30, secondsLen=None, t=None):
    allIndexes = []
    dff = np.abs(np.diff(f))  # Frequency jumps.
    # Indexes of significant frequency jumps.
    nz = np.nonzero(dff > 2)
    nz = np.hstack((0, nz[0], f.size))
    # Windows lengths.
    dnz = np.diff(nz)
    if secondsLen is not None and t is not None:
        samplesNumMin = np.round(secondsLen/(t[1]-t[0]))
    else:
        samplesNumMin = percentLenValid * f.size / 100
    validWinds = np.hstack(np.nonzero(dnz > samplesNumMin))
    fNew = np.zeros_like(f)
    for di in range(validWinds.size):
        currIdxs = nz[validWinds[di]:validWinds[di] + 2]
        allIndexes.append(currIdxs)
        fNew[np.arange(start=currIdxs[0] + 1, stop=currIdxs[1], step=1, dtype='int')] = f[
            np.arange(start=currIdxs[0] + 1, stop=currIdxs[1], step=1, dtype='int')]
    validIdxs = np.nonzero(fNew)  # np.hstack(np.nonzero(fNew))
    samplLen = [chunk[-1] - chunk[0] for chunk in allIndexes]
    idx = np.argmax(np.array(samplLen))
    return allIndexes, idx


def main_test():
    myFile = os.path.join(RootFold, '9.wav')
    Fs, signal = wavfile.read(myFile)
    Fs2 = round(Fs/downsampl)
    realFreq = pandas.read_csv(os.path.join(RootFold, '9.csv')).to_numpy(dtype='float64')
    tachoTime = realFreq[:, 0]
    realFreq = realFreq[:, 1]/2
    txt = [dtm.datetime.fromtimestamp(i) for i in tachoTime]
    txt = [i - txt[0] for i in txt]
    txt = np.array([i.total_seconds() for i in txt])
    t = sm.genTime(signal=signal, Fs=Fs)
    (resampled_x, resampled_t) = scipy.signal.resample(signal, round(len(signal) / downsampl), t=t)
    cutIdxs = np.logical_and(resampled_t >= 17, resampled_t < 29)
    '''''
    resampled_x = resampled_x[cutIdxs]
    resampled_t = resampled_t[cutIdxs]
    resampled_x = resampled_x[1:]
    resampled_t = resampled_t[1:]
    '''
    (representation, t0, fVectNew) =pa.DFTbank(resampled_x, rect=2, level=0.2, Fs=Fs2*10, mirrorLen=0.15, df=0.5,
                                             freqLims=(0, 200), formFactor=32)  # 50, 200  # [fVectIni < 300]
    fig = sm.plotRepresentation(resampled_t, representation, fVectNew, (20, 250))
    fig.axes[0].plot(txt, realFreq*10, color='r')
    kwargs = {'Fs': Fs2*10}
    kwargs.update({'freq': realFreq, 'roughFreqs': (20, 250), 'iterations': 1, 'formFactor': (32, 64)})
    (alpha, f, A, theta, res, coefficient, representationP, fVectNewP) = pa.pronyParamsEst(resampled_x, periodsNum=2, percentOverlap=75, **kwargs)
    fig.axes[0].plot(resampled_t, f)


    '''
    with open('resampledData', "rb") as f:
        kws = dill.load(f)
    resampled_t = kws.get('resampled_t')
    representation = kws.get('representation')
    fVectNew = kws.get('fVectNew')
    f = kws.get('f')
    kwargs = kws.get('kwargs')
    fig = sm.plotRepresentation(resampled_t, representation, fVectNew, (20, 120))
    #fig.axes[0].plot(txt, realFreq*10, color='r')
    fig.axes[0].plot(resampled_t, f)
    '''

    fSmooth = pa.medianSmooth(f, t=resampled_t, secondsWidth=0.5)
    fig.axes[0].plot(resampled_t, fSmooth)
    resampled_x = resampled_x[~np.isnan(fSmooth)]
    resampled_t = resampled_t[~np.isnan(fSmooth)]
    fSmooth = fSmooth[~np.isnan(fSmooth)]
    pass
    shift = fSmooth/fSmooth[0]
    resampledSignal = pa.chirpResample(resampled_x, shift=shift, dt=resampled_t[1]-resampled_t[0])
    figSpec = plt.figure()
    grid = matplotlib.gridspec.GridSpec(1, 1)
    ax_specWav = figSpec.add_subplot(grid[0])
    fVectIni = np.fft.rfftfreq(signal.size, 1 / Fs)
    specIni = np.fft.rfft(signal) / signal.size
    ax_specWav.plot(fVectIni, abs(specIni))
    fVectRes = np.fft.rfftfreq(resampledSignal.size, 1 / Fs2)
    specRes = np.fft.rfft(resampledSignal) / resampledSignal.size
    ax_specWav.plot(fVectRes, abs(specRes))
    (representationR, t0R, fVectNewR) =pa.DFTbank(resampledSignal, rect=2, level=0.2, Fs=Fs2*10, df=0.5, freqLims=(0, 200), formFactor=32)
    kwargs.update({'freq': realFreq, 'roughFreqs': (20, 120), 'iterations': 1, 'formFactor': (32, 64), 'hold': 0})
    '''''
    with open('bufferWave', "rb") as f:
        kws = dill.load(f)
    t0R = kws.get('t0R')
    representationR = kws.get('representationR')
    fVectNewR = kws.get('fVectNewR')
    resampledSignal = kws.get('resampledSignal')
    kwargs = kws.get('kwargs')
    '''
    figR = sm.plotRepresentation(t0R, representationR, fVectNewR, (20, 120))
    (alpha, fR, A, theta, res, coefficient, representationP, fVectNewP) = pa.pronyParamsEst(resampledSignal, periodsNum=10, percentOverlap=75, **kwargs)
    figR.axes[0].plot(t0R[0:len(fR)], fR[0:len(t0R)])
    resampledSignal = resampledSignal[~np.isnan(fR)]
    t0R = t0R[~np.isnan(fR[1:])]
    fR = fR[~np.isnan(fR)]
    shift = fR / fR[0] / 10
    resampledSignal2 = pa.chirpResample(resampledSignal, shift=shift, dt=resampled_t[1] - resampled_t[0])
    (representationR2, t0R2, fVectNewR2) =pa.DFTbank(resampledSignal2, rect=2, level=0.2, Fs=Fs2*10, df=0.5, freqLims=(0, 200), formFactor=32)
    figR2 = sm.plotRepresentation(t0R2, representationR2, fVectNewR2, (20, 120))
    fVectRes2 = np.fft.rfftfreq(resampledSignal2.size, 1 / Fs2)
    specRes2 = np.fft.rfft(resampledSignal2) / resampledSignal2.size
    ax_specWav.plot(fVectRes2, abs(specRes2))

    myFile = os.path.join('Out', 'resampledData', '9.wav')
    scipy.io.wavfile.write(myFile, Fs2, resampledSignal2)
    pass


if __name__ == "__main__":
    main()
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from scipy.signal import hilbert, chirp
import scipy
import pronyAnalysis as pa
import dill
import copy
import os
from multiprocessing import Pool


def waveProc(kwargs):
    signal = kwargs.get('signal')
    SNR = kwargs.get('SNR')
    dFmax = kwargs.get('dFmax')
    Fs = kwargs.get('Fs')
    freq = kwargs.get('freq')
    selectWinds = kwargs.get('selectWinds')
    signalN = pa.awgn(signal, SNRdB=SNR)[0]
    (representation, t0, fVectNew) = pa.DFTbank(signalN, rect=2, mirrorLen=0.15, level=0.2, freqLims=(50, 200), Fs=Fs,
                                                df=1, formFactor=1024)
    '''''
    if plotGraphs == 2:
        fig4 = plt.figure()
        grid = matplotlib.gridspec.GridSpec(1, 1)
        ax_specWav = fig4.add_subplot(grid[0])
        extent = t[0], t[-1], fVectNew[0], fVectNew[-1]
        ax_specWav.imshow(np.flipud(np.abs(representation)), extent=extent)
        ax_specWav.axis('auto')
        plt.ylim(int(np.min(freq) * 0.9), int(np.max(freq) * 1.1))
        fig4.show()
    '''''

    repFreq = pa.representationTrack(representation, fVectNew)

    (representation, t0, fVectNew) = pa.DFTbank(signalN, rect=2, mirrorLen=0.15, level=0.2, freqLims=(50, 200), Fs=Fs,
                                                df=1, formFactor=128)
    (alpha, f, A, theta, resid, coefficient) = pa.thresholdRepreProny(representation, fVectNew, Fs=Fs, centralFreq=0,
                                                                      lowFreq=95, highFreq=130, hold=1.4)
    '''''
    if plotGraphs:
        ax_FreqDev.plot(t, freq[0, :], linestyle='-', color='k', label='Мгновенная частота')
        ax_FreqR.plot(t, freq[0, :], linestyle='-', color='k', label='Мгновенная частота')
    '''''
    indexes = np.hstack(np.nonzero(f))
    # Select long stable windows.
    if selectWinds:
        errIdxs = pa.SM.validateTrack(f)
        errIdxsRepr = pa.SM.validateTrack(repFreq)
    else:
        errIdxs = np.hstack(np.nonzero(f))
        errIdxsRepr = np.hstack(np.nonzero(repFreq))
    validLen = errIdxs.size / f.size
    if validLen < 0.85:
        errIdxs = indexes
    validLenRepr = errIdxsRepr.size / f.size
    if validLenRepr < 0.85:
        errIdxsRepr = indexes
    '''''
    if plotGraphs:
        ax_FreqDev.plot(t[indexes], f[indexes], linestyle=':', color=cmap1(bi), label='Оценка частоты')
        # ax_FreqDev.plot(t[errIdxs], f[errIdxs], linestyle=':', color='r', label='Поправленная оценка')
        ax_FreqDev.set_xlabel('Время, сек')
        ax_FreqDev.set_ylabel('Частота, Гц')
        ax_FreqR.plot(t, repFreq, linestyle=':', color=cmap1(bi), label='Оценка частоты')
        ax_FreqR.set_xlabel('Время, сек')
        ax_FreqR.set_ylabel('Частота, Гц')
    '''''
    diff = freq[0, errIdxs] - f[errIdxs]
    err = pa.rms(diff)
    errRep = pa.rms(freq[0, errIdxsRepr] - repFreq[errIdxsRepr])
    #print('Coefficient {}, error {}, valid track length {} percents, representation error {}, repres. valid track len = {}'.format(
            #dFmax, err, validLen * 100, errRep, validLenRepr * 100))
    '''''
    if ai < 1 and bi < 1 and plotGraphs:
        ax_FreqDev.legend()
        ax_FreqR.legend()
    '''''
    return err, errRep, validLen, validLenRepr


def main():
    plotGraphs=False
    selectWinds=True
    Fs = 9600
    dt = 1/Fs
    t = np.arange(0, 1, dt)
    df = Fs/t.size
    fVect = np.fft.rfftfreq(t.size, 1/Fs) # np.arange(0, Fs/2-df, df)

    dFmax = np.array([0.0, 0.1, 0.2, 0.3])
    experiencies = 102
    fig = []
    figRepr = []
    figDF = plt.figure()
    grid = matplotlib.gridspec.GridSpec(1, 1)
    ax_dF = figDF.add_subplot(grid[0])
    figDFrep = plt.figure()
    gridDFR = matplotlib.gridspec.GridSpec(1, 1)
    ax_dFrepr = figDFrep.add_subplot(gridDFR[0])
    linestyles = ('-', '--', ':', '-.')
    matplotlib.rc('font', family='Times New Roman')
    SNRs = np.arange(start=10, stop=-16.5, step=-0.5, dtype='float64')
    cmap1 = plt.cm.get_cmap('magma', experiencies)
    errAverCoeff = []  # Array by SNR for all frequency increasing coefficients.
    errAverRepCoeff = []

    for ai in range(dFmax.size):
        errAver = []  # Array by SNR for the current frequency increasing coefficient.
        errAverRep = []
        for ci in range(SNRs.size):
            errTot = []  # Array by experiences.
            errTotRep = []  # Array by experiences.
            validLen = []  # Length of valid track related to the common signal length.
            validLenRepr = []
            if plotGraphs:
                fig.append(plt.figure(figsize=(8, 6)))
                grid = matplotlib.gridspec.GridSpec(1, 1)
                ax_FreqDev = fig[-1].add_subplot(grid[0])
                ax_FreqDev.title.set_text('Coefficient = {}, SNR = {}'.format(dFmax[ai], SNRs[ci]))
                ax_FreqDev.grid()
                figRepr.append(plt.figure(figsize=(8, 6)))
                gridR = matplotlib.gridspec.GridSpec(1, 1)
                ax_FreqR = figRepr[-1].add_subplot(gridR[0])
                ax_FreqR.title.set_text('Time-frequency track. Coefficient = {}, SNR = {}'.format(dFmax[ai], SNRs[ci]))
                ax_FreqR.grid()
            coeff = dFmax[ai]/t.size/dt
            freq = np.ones((1, t.size))
            freq[0, :] *= 100 * (1 + np.linspace(start=0, stop=dFmax[ai], num=t.size))
            ampl = np.array([0.7]) #
            decay = np.array([0]) #, 0.7, 7
            phase = np.array([np.pi/4]) #
            signal = ampl[0]*chirp(t, freq[0, 0], t[-1], freq[0, -1])

            kwargs = {'signal': signal, 'SNR': SNRs[ci], 'dFmax': dFmax[ai], 'Fs': Fs, 'freq': freq, 'selectWinds': selectWinds}
            kwCopy = [copy.deepcopy(kwargs) for i in range(experiencies)]
            with Pool(processes=3) as pool:
                result = []
                for r in pool.imap_unordered(waveProc, kwCopy):
                    result.append(r)
            pool.close()
            pool.join()

            for bi in range(experiencies):
                (err, errRep, validLen1, validLenRepr1) = result[bi]  # waveProc(kwargs)
                errTot.append(err)
                errTotRep.append(errRep)
                validLen.append(validLen1)
                validLenRepr.append(validLenRepr1)
            errAver.append(np.mean(np.array(errTot)))
            errAverRep.append(np.mean(np.array(errTotRep)))
            print('Coefficient = {}, SNR = {}, AVERAGE error = {}, AVERAGE valid track len = {}, AVG representation {}, REPR valid track len = {}'.format(dFmax[ai], SNRs[ci], errAver[-1], np.mean(np.array(validLen))*100, errAverRep[-1], np.mean(np.array(validLenRepr))*100))
            if plotGraphs:
                fig[-1].show()
                figRepr[-1].show()
            pass
        errAverArr = np.array(errAver)
        ax_dF.plot(SNRs, errAverArr, linestyle=linestyles[ai], color='k', label=r'$k_i = {}$'.format(dFmax[ai]))
        ax_dF.set_xlabel('ОСШ, дБ')
        ax_dF.set_ylabel('СКО частоты, Гц')
        ax_dF.legend()
        errAverArrRep = np.array(errAverRep)
        ax_dFrepr.plot(SNRs, errAverArrRep, linestyle=linestyles[ai], color='k', label=r'$k_i = {}$'.format(dFmax[ai]))
        ax_dFrepr.set_xlabel('ОСШ, дБ')
        ax_dFrepr.set_ylabel('СКО частоты, Гц')
        ax_dFrepr.legend()
        ax_dFrepr.title.set_text('Time-frequency track')
        errAverRepCoeff.append(errAverArrRep)
        errAverCoeff.append(errAverArr)

    figDF.show()
    figDFrep.show()
    if not (os.path.exists('Out') and os.path.isdir('Out')):
        os.mkdir('Out')
    file_name = 'wavelet128Fine1016_2.pkl'
    with open(os.path.join('Out', file_name), "wb") as f:
        pool = None
        kwCopy = None
        kwSave = {'SNRs': SNRs, 'errAverArr': errAverCoeff, 'errAverRep': errAverRepCoeff}
        dill.dump(kwSave, f)
        f.close()
    pass


if __name__ == "__main__":
    main()

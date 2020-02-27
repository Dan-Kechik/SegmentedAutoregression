import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from scipy.signal import hilbert, chirp
import scipy
import pronyAnalysis as pa

plotGraphs=True
Fs = 9600
dt = 1/Fs
t = np.arange(0, 1, dt)
df = Fs/t.size
fVect = np.fft.rfftfreq(t.size, 1/Fs) # np.arange(0, Fs/2-df, df)

dFmax = 0.1
distF = 5
coeff = dFmax/t.size/dt
freq = np.ones((4, t.size)) #50, 60
freq[0, :] = freq[0, :]*100*(1+np.linspace(start=0, stop=dFmax, num=t.size))
freq[1, :] = freq[0, :]+distF
freq[2, :] = freq[2, :]*125
freq[3, :] = freq[2, :]+distF
ampl = np.array([0.7, 0.5, 1, 1]) #
decay = np.array([0, 0, 0, 0]) #, 0.7, 7
phase = np.array([np.pi/4, np.pi/6, np.pi/4, np.pi/6]) #
signal = np.zeros(t.shape)
k = np.zeros_like(decay)

'''''
comp = chirp(t, freq[0, 0], t[-1], freq[0, -1])
signal = comp + chirp(t, freq[0, 0]+50, t[-1], freq[0, -1]+50)
signal = pa.abwgn(signal, SNRdB=-10, Fs=Fs, band=(freq[0, 0]+45, freq[0, -1]+60))[0]
spec = np.fft.rfft(signal)/signal.size
spec[fVect<freq[0, 0]*0.95] = 0
spec[fVect>freq[0, -1]*1.05] = 0
cmp = np.fft.irfft(spec)*spec.size*2

fig = plt.figure()
plt.plot(t, comp)
plt.plot(t, cmp)
fig.show()
'''''

percentLength = 5
percentOverlap = 50
coefficients = np.array([0, 0.1, 0.2, 0.3])
SNRs = np.arange(start=10, stop=-31, step=-1, dtype='float64')  # np.array([3., 0., -3., -6, -10., -12])
distances = np.arange(start=10, stop=3, step=-1)
experiences=100
totHarmResid = []
totNoisResid = []
totDF = []
totDF2 = []

fig = plt.figure(figsize=(8, 6))  # fig = []  # v0.1
grid1 = matplotlib.gridspec.GridSpec(1, 1)
ax_Resid = fig.add_subplot(grid1[0])
fig2 = plt.figure(figsize=(8, 6))  # fig2 = []
grid2 = matplotlib.gridspec.GridSpec(1, 1)
ax_FreqDev = fig2.add_subplot(grid2[0])
linestyles = ('-', '--', ':', '-.')
matplotlib.rc('font', family='Times New Roman')

for ai in range(coefficients.size):
    harmResid = np.zeros_like(SNRs)
    noisResid = np.zeros_like(SNRs)
    harmDisp = np.zeros_like(SNRs)
    noisDisp = np.zeros_like(SNRs)
    dF = np.zeros_like(SNRs)
    dF2 = np.zeros((SNRs.size, distances.size))
    freq = np.ones((1, t.size))
    freq[0, :] = freq[0, :] * 100 * (1 + np.linspace(start=0, stop=coefficients[ai], num=t.size))
    (frq, frqWinIdxs) = pa.windowing(freq[0, :], int(np.round(freq[0, :].size * percentLength / 100)), percentOverlap)
    fOrig = frq[:, 0]
    for bi in range (SNRs.size):
        for ci in range(experiences):
            signal = chirp(t, freq[0, 0], t[-1], freq[0, -1])
            #(alpha, f, A, theta, resid, coefficient) = pa.timeSegmentedProny(signal, Fs=Fs, percentLength=10, percentOverlap=50, order=2)
            (signal, noise) = pa.awgn(signal, SNRdB=SNRs[bi])[0:2]
            spec = np.fft.rfft(signal) / signal.size
            spec[fVect < 95] = 0  # freq[0, 0] * 0.95
            spec[fVect > 135] = 0  # freq[0, -1] * 1.05
            cmp = np.fft.irfft(spec) * spec.size * 2
            (alpha, f, A, theta, resid, coefficient) = pa.timeSegmentedProny(cmp, Fs=Fs, percentLength=percentLength, percentOverlap=percentOverlap, order=2)
            fTemp = np.array(f)[:, 1]
            dF[bi] += pa.rms(fOrig-fTemp)
            harmDisp += np.std(fTemp)
            spec2 = np.fft.rfft(noise) / noise.size
            spec2[fVect < 95] = 0  # freq[0, 0] * 0.95
            spec2[fVect > 135] = 0  # freq[0, -1] * 1.05
            cmp2 = np.fft.irfft(spec2) * spec2.size * 2
            (alpha1, f1, A1, theta1, resid1, coefficient1) = pa.timeSegmentedProny(cmp2, Fs=Fs, percentLength=percentLength, percentOverlap=percentOverlap, order=2)
            harmResid[bi] += np.mean(resid)
            noisResid[bi] += np.mean(resid1)
            fTemp = np.array(f1)[:, 1]
            noisDisp += np.std(fTemp)
            '''''
            for di in range(distances.size):
                mixt = cmp + chirp(t, freq[0, 0]+distances[di], t[-1], freq[0, -1]+distances[di])
                (alpha2, f2, A2, theta2, resid2, coefficient2) = pa.timeSegmentedProny(cmp, Fs=Fs,
                                                                                 percentLength=percentLength,
                                                                                 percentOverlap=percentOverlap, order=4)
                fTemp1 = np.array(f2)[:, 1]
                fTemp2 = np.array(f2)[:, 3]
                dF2[bi, di] += np.maximum(pa.rms(fOrig - fTemp2), pa.rms(fOrig+distances[di] - fTemp1))
                '''''
            '''''
            (representation, t, fVectNew) = pa.DFTbank(signal, rect=1, level=0.2, Fs=Fs, df=1, formFactor=1024)
            (alpha, f, A, theta, resid, coefficient, errors) = pa.representProny(signal, representation, fVectNew, Fs=Fs, percentLength=10, percentOverlap=50, lowFreq=100, highFreq=100, level=0.2)
            # Get in-range errors and frequencies.
            psi = pa.windMEXH(100, t, 1024)[0]
            supportFreqs = pa.wavSpectrum(psi, rect=False, level=0.5, fVectIni=fVectNew, Fs=Fs)[3]
            supportFreqs += 100
            '''''

            ''''
            supportFreqs = np.array([100])
            inRangeIdxs = np.logical_and(freq[0, :]>supportFreqs[0], freq[0, :]<supportFreqs[-1])
            if any(inRangeIdxs):
                fr = range(np.nonzero(inRangeIdxs)[0][0], np.nonzero(inRangeIdxs)[0][-1])
                fInRan = f[fr[0]:fr[-1]]
                residInRan = resid[fr[0]:fr[-1]]
                #coefficientInRan = coefficient[fr[0]:fr[-1]]
                #errorsInRan = errors[fr[0]:fr[-1]]
            spec = np.fft.rfft(signal) / signal.size
            spec[fVect < freq[0, 0] * 1.05] = 0
            spec[fVect > freq[0, -1] * 1.15] = 0
            cmp = np.fft.irfft(spec) * spec.size * 2
            (alpha1, f1, A1, theta1, resid1, coefficient1) = pa.timeSegmentedProny(cmp, Fs=Fs, percentLength=10, percentOverlap=50, order=2)
            psi1 = pa.windMEXH(105, t, 1024, carr=True)[0]
            comp105 = np.convolve(signal, psi1, 'same')
            fig = plt.figure()
            plt.plot(t, psi1)
            plt.plot(t, comp105)
            fig.show()
            (alpha105, f105, A105, theta105, resid105, coefficient105) = pa.timeSegmentedProny(comp105, Fs=Fs, percentLength=10, percentOverlap=50, order=2)
            outRangeIdxs = np.logical_or(freq[0, :]<supportFreqs[0], freq[0, :]>supportFreqs[-1])
            if any(outRangeIdxs):
                fr = range(np.nonzero(outRangeIdxs)[0][0], np.nonzero(outRangeIdxs)[0][-1])
                fOutRan = f1[fr[0]:fr[-1]]
                residOutRan = resid1[fr[0]:fr[-1]]
                #coefficientOutRan = coefficient[fr[0]:fr[-1]]
                #errorsOutRan = errors[fr[0]:fr[-1]]
            '''''
            pass
        harmResid[bi] /= experiences
        noisResid[bi] /= experiences
        dF[bi] /= experiences
        dF2[bi] /= experiences
        pass
    totHarmResid.append(harmResid)
    totNoisResid.append(noisResid)
    totDF.append(dF)
    totDF2.append(dF2)

    ax_Resid.plot(SNRs, harmResid, linestyle=linestyles[ai], color='k', label=r'$k_i = {}$'.format(coefficients[ai]))
    ax_Resid.set_xlabel('ОСШ, дБ')
    ax_Resid.set_ylabel('Ошибки аппроксимации')
    ax_Resid.legend()
    ax_FreqDev.plot(SNRs, dF, linestyle=linestyles[ai], color='k', label=r'$k_i = {}$'.format(coefficients[ai]))
    ax_FreqDev.set_xlabel('ОСШ, дБ')
    ax_FreqDev.set_ylabel('СКО оценки частоты, Гц')
    ax_FreqDev.legend()
    fig.show()
    fig2.show()
    pass
    '''''
    # v0.1
    fig.append(plt.figure())
    plt.plot(SNRs, harmResid/noisResid)
    fig[ai].show()
    plt.xlabel('ОСШ, дБ')
    plt.ylabel('Отношение остатков модели, раз')
    plt.title('Коэффициент нарастания частоты {}'.format(coefficients[ai]))
    fig2.append(plt.figure())
    plt.plot(SNRs, dF)
    fig2[ai].show()
    plt.xlabel('ОСШ, дБ')
    plt.ylabel('Отклонение оценки частоты, Гц')
    plt.title('Коэффициент нарастания частоты {}'.format(coefficients[ai]))
    '''''
    '''''
    fig3 = plt.figure()
    for di in range(distances.size):
        plt.plot(SNRs, dF2[:, di])
    fig3.show()
    plt.xlabel('ОСШ, дБ')
    plt.ylabel('Отклонение оценки частоты для двух близких компонент, Гц')
    plt.title('Коэффициент нарастания частоты {}'.format(coefficients[ai]))
    '''''

noiseResids = np.mean(np.hstack(totNoisResid))
print('Noise residues: {}'.format(totNoisResid))
pass

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from scipy.signal import hilbert, chirp
import scipy
import pronyAnalysis as pa
import dill
import os

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
SNRs = np.arange(start=10, stop=-16.5, step=-0.5, dtype='float64')  # np.array([3., 0., -3., -6, -10., -12])
distances = np.arange(start=10, stop=3, step=-1)
experiences=100
totHarmResid = []
totNoisResid = []
totDF = []
totDF2 = []

fig = plt.figure(figsize=(8, 6))  # fig = []  # v0.1
grid1 = matplotlib.gridspec.GridSpec(1, 1)
ax_Resid = fig.add_subplot(grid1[0])
ax_Resid.grid()
ax_Resid.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
fig2 = plt.figure(figsize=(8, 6))  # fig2 = []
grid2 = matplotlib.gridspec.GridSpec(1, 1)
ax_FreqDev = fig2.add_subplot(grid2[0])
ax_FreqDev.grid()
figPow = plt.figure(figsize=(8, 6))
gridP = matplotlib.gridspec.GridSpec(2, 1)
ax_harmPow = figPow.add_subplot(gridP[0])
ax_harmPow.grid()
ax_harmPow.set_xlabel('Мощность смеси сигнала и шума')
ax_harmPow.set_ylabel('Ошибка аппроксимации')
ax_harmPow.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
ax_noisPow = figPow.add_subplot(gridP[1])
ax_noisPow.set_xlabel('Мощность шума')
ax_noisPow.set_ylabel('Ошибка аппроксимации')
ax_noisPow.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
ax_noisPow.grid()
linestyles = ('-', '--', ':', '-.')
matplotlib.rc('font', family='Times New Roman')

for ai in range(coefficients.size):
    harmResid = np.zeros_like(SNRs)
    noisResid = np.zeros_like(SNRs)
    harmPower = np.zeros_like(SNRs)
    noisPower = np.zeros_like(SNRs)
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
            harmPower[bi] += np.sum(cmp**2)/cmp.size
            noisPower[bi] += np.sum(cmp2**2)/cmp2.size
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
        harmPower[bi] /= experiences
        noisPower[bi] /= experiences
        dF[bi] /= experiences
        dF2[bi] /= experiences
        pass
    totHarmResid.append(harmResid)
    totNoisResid.append(noisResid)
    totDF.append(dF)
    totDF2.append(dF2)

    ax_Resid.plot(SNRs, harmResid, linestyle=linestyles[ai], color='k', label=r'$k_i = {}$'.format(coefficients[ai]))
    ax_Resid.plot(SNRs, noisResid, linestyle=linestyles[ai], color='r', label=r'$k_i = {}$'.format(coefficients[ai]))
    ax_Resid.set_xlabel('ОСШ, дБ')
    ax_Resid.set_ylabel('Ошибки аппроксимации')
    ax_Resid.legend()
    ax_FreqDev.plot(SNRs, dF, linestyle=linestyles[ai], color='k', label=r'$k_i = {}$'.format(coefficients[ai]))
    ax_FreqDev.set_xlabel('ОСШ, дБ')
    ax_FreqDev.set_ylabel('СКО оценки частоты, Гц')
    ax_FreqDev.legend()
    fig.show()
    fig2.show()
    ax_harmPow.plot(harmPower, harmResid, linestyle=linestyles[ai], color='k', label=r'$k_i = {}$'.format(coefficients[ai]))
    ax_noisPow.plot(noisPower, noisResid, linestyle=linestyles[ai], color='k', label=r'$k_i = {}$'.format(coefficients[ai]))
    ax_harmPow.legend()
    ax_noisPow.legend()
    figPow.show()
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
print('Noise residues: {}'.format(noiseResids))

figDff = plt.figure()
gridDff = matplotlib.gridspec.GridSpec(1, 1)
ax_Dff = figDff.add_subplot(gridDff[0])
ax_Dff.grid()
ax_Dff.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
ax_Dff.plot(harmPower, noisResid-harmResid, linestyle='-', color='k')
ax_Dff.set_xlabel('Мощность')
ax_Dff.set_ylabel('Разность ошибкок аппроксимации смеси сигнал+шум и шума')
figDff2 = plt.figure()
gridDff2 = matplotlib.gridspec.GridSpec(1, 1)
ax_Dff2 = figDff2.add_subplot(gridDff2[0])
ax_Dff2.grid()
ax_Dff2.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
ax_Dff2.plot(SNRs, noisResid-harmResid, linestyle='-', color='k')
ax_Dff2.set_xlabel('ОСШ, дБ')
ax_Dff2.set_ylabel('Разность ошибкок аппроксимации смеси сигнал+шум и шума')
figDff3 = plt.figure()
gridDff3 = matplotlib.gridspec.GridSpec(1, 1)
ax_Dff3 = figDff3.add_subplot(gridDff3[0])
ax_Dff3.grid()
ax_Dff3.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
ax_Dff3.plot(SNRs, (noisResid-harmResid)/harmResid, linestyle='-', color='k')
ax_Dff3.set_xlabel('ОСШ, дБ')
ax_Dff3.set_ylabel('Относительная разность ошибкок аппроксимации смеси сигнал+шум и шума')

figFinal = plt.figure()
gridFn = matplotlib.gridspec.GridSpec(1, 1)
ax_Fin = figFinal.add_subplot(gridFn[0])
ax_Fin.grid()
ax_Fin.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
experiences=1
SNRs = np.hstack((np.inf, SNRs, -np.inf))
cmap1 = plt.cm.get_cmap('hsv', SNRs.size+1)
for di in range(SNRs.size):
    cleanPower = np.zeros_like(harmPower)
    cleanResid = np.zeros_like(harmPower)
    for ci in range(experiences):
        signalClean = chirp(t, freq[0, 0], t[-1], freq[0, -1])
        if SNRs[di] == np.inf:
            signal = signalClean
        elif SNRs[di] == -np.inf:
            signal = noise
        else:
            (signal, noise) = pa.awgn(signalClean, SNRdB=SNRs[di])[0:2]
        for bi in range(harmPower.size):
            mult = harmPower[bi]/(np.sum(signal**2)/signal.size)
            signal = signal*np.sqrt(mult)
            cleanPower[bi] = np.sum(signal**2)/signal.size
            (alpha1, f1, A1, theta1, resid1, coefficient1) = pa.timeSegmentedProny(signal, Fs=Fs, percentLength=percentLength,
                                                                                   percentOverlap=percentOverlap, order=2)
            cleanResid[bi] += np.mean(resid1)
    cleanResid /= experiences
    if np.round(SNRs[di]/5) == SNRs[di]/5:
        ax_Fin.plot(cleanPower, cleanResid, linestyle='-', color=cmap1(di), label='SNR {}'.format(SNRs[di]))
#ax_Fin.plot(cleanPower, cleanResid, linestyle='--', color='k', label='Чистый сигнал')
#ax_Fin.plot(harmPower, harmResid, linestyle='-', color='k', label='Сигнал+шум')
#ax_Fin.plot(noisPower, noisResid, linestyle=':', color='r', label="Шум")
ax_Fin.legend()
ax_Fin.set_xlabel('Мощность')
ax_Fin.set_ylabel('Ошибка аппроксимации')

file_name = 'Out\\resolutionTestFine1016.pkl'
if not (os.path.exists('Out') and os.path.isdir('Out')):
    os.mkdir('Out')
dill.dump_session(file_name)
pass

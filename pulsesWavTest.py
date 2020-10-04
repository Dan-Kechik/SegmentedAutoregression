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

ranges = [(500, 680)]
RootFold = r'D:\Repository\ComputeFramework\Classifier\In'
downsampl = 50
processesNumber = 0

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
    print('========='+currentFile+'=========\n\n')
    Fs, signal = wavfile.read(myFile)
    Fs2 = round(Fs/downsampl)
    t = sm.genTime(signal=signal, Fs=Fs)
    (resampled_x, resampled_t) = scipy.signal.resample(signal, round(len(signal) / downsampl), t=t)
    for ran in ranges:
        pulsesSequence = pa.fftFiltrate(resampled_x, ran, Fs=Fs2)[1]
        sm.plotSignal(pulsesSequence, t=resampled_t, specKind='amplitude')
        (alpha, f, A, theta, resid, timeSamples, coefficient, representation, fVectNew) = sm.pulsesParamsEst(pulsesSequence, roughFreqs=ran, Fs=Fs2)
        pass



if __name__ == "__main__":
    main()
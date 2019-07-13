import glob
import os
import numpy as np
import cmath
from scipy import signal
from scipy.io import wavfile

def readNpy(path):
    listFakeA = glob.glob(path + "/*fake_A.npy")
    listFakeB = glob.glob(path + "/*fake_B.npy")
    listRealA = glob.glob(path + "/*real_A.npy")
    listRealB = glob.glob(path + "/*real_B.npy")
    listReconstA = glob.glob(path + "/*rec_A.npy")
    listReconstB = glob.glob(path + "/*rec_B.npy")

    listFakeA.sort()
    listFakeB.sort()
    listRealA.sort()
    listRealB.sort()
    listReconstA.sort()
    listReconstB.sort()

    specFakeA = [np.load(x) for x in listFakeA]
    specFakeB = [np.load(x) for x in listFakeB]
    specRealA = [np.load(x) for x in listRealA]
    specRealB = [np.load(x) for x in listRealB]
    specReconstA = [np.load(x) for x in listReconstA]
    specReconstB = [np.load(x) for x in listReconstB]

    return specFakeA, specFakeB, specRealA, specRealB, specReconstA, specReconstB

def spec2wav(path, sample_rate=44100):
    specFakeA, specFakeB, specRealA, specRealB, specReconstA, specReconstB = readNpy(path)

    ret_wavs = []

    for specs, name in zip([specRealA, specFakeB, specReconstA, specRealB, specFakeA, specReconstB],
                           ["realA", "fakeB", "reconstA",
                            "realB", "fakeA", "reconstB"]):
        sample = np.zeros((len(specs) + 1) * sample_rate // 2)
        for i, spec in enumerate(specs):
            Amp = np.maximum(np.exp((spec[0] + 0.1) * 10) - 1, 0)
            Angle = spec[1] * np.pi
            spectrogram = Amp * (np.cos(Angle) + cmath.sqrt(-1) * np.sin(Angle))
            t_istft, x_istft = signal.istft(spectrogram, sample_rate, nperseg=346, nfft=510)
            sample[i * sample_rate // 2:i * sample_rate // 2 + sample_rate] += x_istft[:sample_rate]
        sample[sample_rate // 2:len(sample) - sample_rate // 2] /= 2
        # wavfile.write(path + name + '.wav', sample_rate, sample.astype(np.int16))
        ret_wavs.append(sample.astype(np.int16))
    return ret_wavs


def spec2img(path):
    specFakeA, specFakeB, specRealA, specRealB, specReconstA, specReconstB = readNpy(path)

    amps = []
    angles = []

    for specs in [specRealA, specFakeB, specReconstA, specRealB, specFakeA, specReconstB]:
        amps.append(((specs[0][0] + 1) / 2.0 * 255.0).astype(np.uint8))
        angles.append(((specs[0][1] + 1) / 2.0 * 255.0).astype(np.uint8))

    return amps, angles
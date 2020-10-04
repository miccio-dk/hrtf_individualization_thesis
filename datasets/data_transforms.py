import numpy as np
import torch
import scipy.fft as fft
import scipy.signal as sig
from scipy.interpolate import interp1d


# convert hrir to hrtf
class ToHrtf(object):
    def __init__(self, nfft):
        self.nfft = nfft

    def __call__(self, sample):
        hrir, labels = sample
        hrtf = fft.rfft(hrir, self.nfft)
        hrtf_abs = np.abs(hrtf)
        return (hrtf_abs, labels)


# extract spectral envelope from hrtfs
class SpecEnv(object):
    def __init__(self, nfft, feature='specenv', cutoff=None):
        self.f = fft.rfftfreq(nfft, 1. / 2)
        self.cutoff = cutoff
        self.feature_fnc = {
            'specenv': self.specenv,
            'notches': self.notches
        }.get(feature)
        self.eps = np.finfo(np.float32).eps

    def __call__(self, sample):
        hrtf, labels = sample
        spec_feature = self.feature_fnc(hrtf)
        return (spec_feature, labels)

    def specenv(self, s):
        s += 1
        # find peaks and peak frequencies
        locs, _ = sig.find_peaks(s + self.eps)
        peaks = s[locs]
        fpeaks = self.f[locs]
        # add extremes to fpeaks and peaks
        peaks = np.array([s[0], *peaks, s[-1]])
        fpeaks = np.array([self.f[0], *fpeaks, self.f[-1]])
        # interpolate across peaks
        fnc = interp1d(fpeaks, peaks, kind='cubic')
        specenv = fnc(self.f)
        # fade envelope into spectrum after cutoff
        if self.cutoff is not None:
            specenv[self.f > self.cutoff] = s[self.f > self.cutoff]
        return specenv

    def notches(self, s):
        specenv = self.specenv(s)
        notches = s - specenv + 1
        return notches


# convert numpy array to pytorch tensor
class ToTensor(object):
    def __init__(self, use_float=True):
        self.use_float = use_float

    def __call__(self, sample):
        x, labels = sample
        x = torch.from_numpy(x)
        if self.use_float:
            x = x.float()
        return (x, labels)

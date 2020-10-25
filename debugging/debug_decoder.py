import json
import torch
import numpy as np
import pandas as pd
import librosa as lr
import scipy.signal as sig
import matplotlib.pyplot as plt

from ..models.decoders import DecoderCfg
from ..models.layers import IIRPeakTaylor, IIRFilterResponse

# data to test
peak0_w0 = torch.Tensor([[0.0]])
peak0_bw = torch.Tensor([[0.0]])
peak0_g = torch.Tensor([[-0.84154165]])
peak1_w0 = torch.Tensor([[-0.02203104]])
peak1_bw = torch.Tensor([[-0.5523975]])
peak1_g = torch.Tensor([[-0.16775638]])
z = ((peak0_w0, peak0_bw, peak0_g), (peak1_w0, peak1_bw, peak1_g))

## 1. PREDICT USING DECODER
# model setup
nfft = 256
cfg_path = 'lightning/configs/specenv/small.json'
# load cfg
with open(cfg_path) as fp:
    cfg = json.load(fp)
# init model
dec = DecoderCfg(nfft=nfft, cfg=cfg)
dec.eval()
# predict
resp1 = dec(*z)
print(resp1)
#plt.plot(resp1.T)

### 2. PREDICT DIRECTLY
# init layers
iir_peak = IIRPeakTaylor()
iir_resp = IIRFilterResponse(nfft=nfft)
# forward step
b, a = iir_peak(*z[0])
resp2 = iir_resp(b, a)
print(resp2)
#plt.plot(resp2)

### 3. HOW DOES SCIPY DO IT?
h, w = sig.freqz(b.numpy()[0], a.numpy()[0], worN=128)
print(h)

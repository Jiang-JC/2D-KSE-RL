import numpy as np

def PSD(data):
    data = np.squeeze(data)
    ps = np.abs(np.fft.fft2(data))
    return ps[0][1], ps[1][1], ps[1][0]

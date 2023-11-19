Gaussian average filtering decomposition (GAFD) for signal S.

Please cite the following reference: 

Yue-Der Lin*, Yong Kok Tan and Baofeng Tian, "A novel approach for decomposition of biomedical signals in different applications based on data-adaptive Gaussian average filtering", Biomedical Signal Processing and Control, Vol.71, Part A, 103104, January 2022. (https://doi.org/10.1016/J.BSPC.2021.103104)
Yue-Der Lin*, Yong-Kok Tan, Tienhsiung Ku and Baofeng Tian, "A frequency estimation scheme based on Gaussian average filtering decomposition and Hilbert transform: with estimation of respiratory rate as an example", Sensors, Vol.23, No.8, 3785, April 2023. (https://doi.org/10.3390/s23083785) 


Usage examples:

1) For LOD.txt:

import numpy as np

import matplotlib.pyplot as plt

LOD = np.loadtxt('LOD.txt', dtype=np.float64, delimiter=',')

from gafd import gafd

imf, res = gafd(LOD, 4, 2, 1.6, 'd', 'd', 20, 0.001, True)

2) For fingerPPGwithRIIV.mat:

import scipy.io as IO

S = IO.loadmat('fingerPPGwithRIIV.mat') 

resp = S['resp'].ravel(); ppg = S['ppg'].ravel()

from gafd import gafd

imf, res = gafd(ppg, 6, 2, 1.8, 'd', 'd', 20, 0.001, True)

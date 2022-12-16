Gaussian average filtering decomposition (GAFD) for signal S.
Please cite the following reference: 

Yue-Der Lin*, Yong Kok Tan and Baofeng Tian, "A novel approach for decomposition of biomedical signals in different applications based on data-adaptive Gaussian average filtering", Biomedical Signal Processing and Control, Vol.71, Part A, 103104, January 2022.

Reference: https://doi.org/10.1016/J.BSPC.2021.103104

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

imf, res = gafd(ppg, 6, 2, 1.6, 'd', 'd', 20, 0.001, True)

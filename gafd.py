def gafd(S, nIMF=2, mMask=2, chi=1.6, eType1='d', eType2='d', \
    TH1=20.0, TH2=0.001, sFig=True):
     '''
     Gaussian average filtering decomposition (GAFD) for signal S.
     Designed by Yue-Der Lin, Feng Chia University, Taiwan.

     Reference: https://doi.org/10.1016/J.BSPC.2021.103104

     Inputs:
          S    The signal to be decomposed (an array).
       nIMF    Number of IMF expected to decompose (default = 2).
      mMask    mMask = 1 for mask = int(chi*np.max((np.min(np.diff(Max)),
                                                          np.min(np.diff(Min)))),
               mMask = 2 for mask = int(2*np.floor(chi*L/nMaxMin))
               (default = 2).
        chi    The parameter that influence the window length (default = 1.6).
               chi is between 1.1 and 3 if mMask = 2 (can start from 1.6).
               For mMask = 1, chi is suggested to try from 2.
     eType1    The extention type for the left boundary point:
               'p' (periodical) repeats the pattern outside the boundaries,
               'c' (constant) extends outside the boundaries with the last
                   values achieved at the boundaries (default),
              'r' (reflection) extends the signal symmetrical with respect
                  to the vertical lines over the boundary points.
               'd' (double-symmetric reflection) extends the signal firstly
                   symmetrical with respect to the vertical lines over the
                   boundary point and and next symmetrical with respect to
                   horizontal line over the boundary point.
     eType2    The extention type for the right boundary point:
               'p' (periodical) repeats the pattern outside the boundaries,
               'c' (constant) extends outside the boundaries with the last
                   values achieved at the boundaries (default),
               'r' (reflection) extends the signal symmetrical with respect
                   to the vertical lines over the boundary points.
               'd' (double-symmetric reflection) extends the signal firstly
                   symmetrical with respect to the vertical lines over the
                   boundary point and and next symmetrical with respect to
                   horizontal line over the boundary point.
        TH1    Threshold value for signal to residual energy ratio,
               the 1st decomposition stop criterion (default = 20), and is
               computed as 10*log10(||S(t)||^2/||S_k(t)||^2), where
                           S(t) is the original signal,
                           S_k(t) is the residual of the kth IMF.
        TH2    Threshold value for convergence check,
               the 2nd decomposition stop criterion (default = 0.001), and is
               computed as ||{S_{i-1}(t)-S_{i}(t)||^2/||S_{i}(t)||^2 at i-th
               iteration.
       sFig    True = Show the figures, False = Do not show the figures
               (default = True).

     Output:
        imf    Matrix containg all of the IMF, an array of (number of IMF, length of S).
        res    S - (summation of imf), an array of (length of S, ).

     Usage example:
        import numpy as np
        import matplotlib.pyplot as plt
        LOD = np.loadtxt('LOD.txt', dtype=np.float64, delimiter=',')
        plt.plot(LOD)
        from gafd import gafd
        imf, res = gafd(LOD, 4, 2, 1.6, 'd', 'd', 20, 0.001, True)
     '''
     # Import modules:
     import matplotlib
     import matplotlib.pyplot as plt
     import numpy as np
     from numpy import linalg as LA
     from scipy import signal
     from scipy import ndimage
     from scipy.ndimage import gaussian_filter1d
     import math

     # Initialization:
     L = np.size(S)
     S1 = S
     imf = np.empty(L)
     matplotlib.rcParams['font.family'] = 'Times'

     #   energyRatio: pre-defined threshold for signal to residual energy ratio.
     energyRatio = 10   # Sifting as energyRatio < TH1:
     #   rTolerancxe: pre-defined threshold for convergence check.
     rTolerance = 1     # Sifting as rTolerance > TH2:

     # GAFD:
     #   ind: index for IMF matrix, indicating the ind-th row.
     ind = 0
     while (ind < nIMF):
        energyRatio = 10*math.log10(LA.norm(S,2)/LA.norm(S1,2))
        MaxMin, Max, Min =  numExtrema(S1)
        nMaxMin = np.size(MaxMin)
        # from scipy.signal import argrelmax
        # from scipy.signal import argrelmin
        # Max = argrelmax(S1)
        # nMax = np.size(Max)
        # Min = argrelmin(S1)
        # nMin = np.size(Min)
        # nMaxMin = nMax + nMin

        if (energyRatio > TH1) or (nMaxMin <= 2):
            break
        else:
            if mMask == 1:
                # For mMask=1, the parameter chi is suggested to be 2.
                # mask = int(chi*np.max((np.max(np.diff(Max)), np.max(np.diff(Min)))))
                mask = int(chi*np.max((np.min(np.diff(Max)), np.min(np.diff(Min)))))
            else:
                # For mMask=2, the parameter chi can be a number within 1.1 and 3.
                mask = int(2*np.floor(chi*L/nMaxMin))

        # Sifting process initialization:
        S2 = S1

        # Sifting process:
        rsigPrev = S2

        if (nMaxMin > 2) and (mask < L/2.0):
            gwLength = 2*mask+1
            H = signal.windows.gaussian(gwLength, std=mask/4.0728, sym=True)
            W = H/np.sum(H)

            # Generate new pattern according to eType1 and eType2:
            # ### For left boundary point:
            if eType1 == 'p':
                St_tmp = np.concatenate((S2[-(mask+1):-1], S2), axis=None)
            elif eType1 == 'r':
                St_tmp = np.concatenate((S2[mask:0:-1], S2), axis=None)
            elif eType1 == 'c':
                St_tmp = np.concatenate((S2[0]*np.ones((mask)), S2), axis=None)
            else: # eType == 'd'
                xt1 = 2*S2.mean() - S2[mask:0:-1]
                St_tmp = np.concatenate((xt1, S2), axis=None)

            ### For right boundary point:
            if eType2 == 'p':
                St = np.concatenate((St_tmp, S2[1:mask+1]), axis=None)
            elif eType2 == 'r':
                St = np.concatenate((St_tmp, S2[-2:-(mask+2):-1]), axis=None)
            elif eType2 == 'c':
                St = np.concatenate((St_tmp, S2[-1]*np.ones((mask))), axis=None)
            else: # eType2 == 'd'
                xt2 = 2*S2.mean() - S2[-2:-(mask+2):-1]
                St = np.concatenate((St_tmp, xt2), axis=None)

            # Filtering:
            ave = np.empty((L))
            for i in range(L):
                ave[i] = np.dot(W, St[i:i+gwLength])

        else:
            break

        if sFig == True:
            # Plot the iterative procedure for each imf:
            l = np.arange(L)
            fig1 = plt.figure(1, figsize = (7.2, 5.4))
            # plt.suptitle('Signal and Average')
            plt.subplot(nIMF, 1, ind+1)
            plt.plot(l, S2, l, ave)
            plt.xlim(0, L)
            plt.ylabel('Step {}'.format(ind+1))
            fig1.subplots_adjust(hspace = 1.0)

            ### For last sub-figure:
            if ind == nIMF-1:
                plt.xlabel('Samples')
            else:
                pass

            ### Save the figure:
            plt.savefig('Fig_Signal_Ave.jpg', dpi=1200, transparent=True)
        else:
            pass

        # The resulted S2 is one imf.
        S2 = S2 - ave

        # Residual tolerance:
        rTolerance = (LA.norm(rsigPrev-S2,2)/LA.norm(S1,2))**2

        # The second convergence check:
        if (rTolerance < TH2):
            break

        if ind == 0:
            imf = S2
        else:
            imf = np.vstack((imf,S2))

        S1 = S1 - S2

        ind = ind + 1

     if imf.ndim > 1:
        (m,_) = imf.shape
        res = S

        for i in range(m):
            res = res - imf[i,:]

            if sFig == True:
               # Plot each imf:
               fig2 = plt.figure(2, figsize = (7.2, 5.4))
               # plt.suptitle('IMFs')
               plt.subplot(m,1,i+1)
               plt.plot(imf[i,:])
               plt.xlim(0, L)
               plt.ylabel('IMF {}'.format(i+1))
               fig2.subplots_adjust(hspace = 0.9)

               ### For last sub-figure:
               if i == m-1:
                  plt.xlabel('Samples')
               else:
                  pass

               ### Save the figure:
               plt.savefig('Fig_imf.jpg', dpi=1200, transparent=True)
            else:
               pass

        if sFig == True:
           # Plot signal S and the residual:
           l = np.arange(L)
           plt.figure(3, figsize = (7.2, 5.4))
           plt.plot(l, S, label='Signal')
           plt.plot(l, res, label='Residue')
           plt.xlabel('Samples', fontsize = 13)
           plt.xlim(0, L)
           plt.legend(loc=3, fontsize = 11)
           plt.savefig('Fig_Signal_Res.jpg', dpi=1200, transparent=True)
        else:
           pass

        return imf, res

     else:
        imf = S
        res = np.zeros(L)
        if sFig == True:
           plt.figure(figsize = (7.2, 5.4))
           plt.plot(imf, label='Signal')
           plt.plot(res, label='Residue')
           plt.xlabel('Samples', fontsize = 13)
           plt.xlim(0, L)
           plt.legend(loc=3, fontsize = 11)
           plt.savefig('Fig_Signal_Res.jpg', dpi=1200, transparent=True)
           print('Signal is a simple mode, decomposition is not necessary.')
        else:
           pass

        return imf, res

def numExtrema(S):
    '''
    Find the indices of extremas in signal S.

    Input:
        S    The signal to be decomposed (an array).

  Outputs:
   MaxMin    Array for the indices of extrema points.
      Max    Array for the indices of maxima points.
      Min    Array for the indices of minima points.
    '''

    import numpy as np

    # Find the indices of maxima points:
    D1 = np.where(np.diff(S)>0, 1, 0)
    D2 = np.where(np.diff(D1)<0, 1, 0)
    Max = np.argwhere(D2).ravel()+1

    # Find the indices of minima points:
    X = -S
    D3 = np.where(np.diff(X)>0, 1, 0)
    D4 = np.where(np.diff(D3)<0, 1, 0)
    Min = np.argwhere(D4).ravel()+1

    # Find the indices of extrema points:

    MaxMin = np.sort(np.concatenate((Max, Min), axis=0))

    return MaxMin, Max, Min

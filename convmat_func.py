import numpy as np

def convmat(A, P, Q = 1):
    Nx = A.shape[0]
    Ny = A.shape[1]
    assert(P <= Nx), 'Cannot have more Fourier pts than real-space pts'
    assert(Q <= Ny), 'Cannot have more Fourier pts than real-space pts'
        
    # comp. indices of spatial harmonics
    Nharmonics = P*Q
    p = range(-int(np.floor(P/2)), int(np.floor(P/2))+1)
    q = range(-int(np.floor(Q/2)), int(np.floor(Q/2))+1)
    
    # do fft
    A = np.fft.fft2(A)/(Nx*Ny) # TODO change depending on 1D or 2D
    A = np.fft.fftshift(A) 
    
    # locate zeroth harmonics
    p0 = int(np.floor(Nx/2))
    q0 = int(np.floor(Ny/2))
    
    # calc the convolutoiin matrix
    ret = np.zeros((Nharmonics, Nharmonics), dtype = complex)
    # TODO vectorize
    for qrow in range(0, Q):
        for prow in range(0, P):
            row = (qrow)*P + prow
            for qcol in range(0, Q):
                for pcol in range(0, P):
                    col = (qcol)*P + pcol
                    pfft = p[prow] - p[pcol]
                    qfft = q[qrow] - q[qcol]
                    ret[row, col] = A[p0+pfft, q0+qfft]
    return ret

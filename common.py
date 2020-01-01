import numpy as np

def convmat(A, P, Q = 1):
    Nx = A.shape[0]
    Ny = A.shape[1]
    assert(P <= Nx and Q <= Ny), 'Cannot have more Fourier pts than real-space pts'
        
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

def matmul(*args):
    if len(args) == 1 or len(args) == 0:
        raise ValueError('Need at least two args')
    count = 0
    ret = None
    for arg in args:
        if count == 0:
            ret = arg
        else:
            ret = np.matmul(ret, arg)
        count += 1
    return ret

def redheffer_star_prod(sa_mat, sb_mat, unit_mat):
    Nharm = int(sa_mat.shape[0]/4)
    sa_11_mat = sa_mat[0:2*Nharm, 0:2*Nharm]
    sa_12_mat = sa_mat[0:2*Nharm, 2*Nharm:4*Nharm]
    sa_21_mat = sa_mat[2*Nharm:4*Nharm, 0:2*Nharm]
    sa_22_mat = sa_mat[2*Nharm:4*Nharm, 2*Nharm:4*Nharm]

    sb_11_mat = sb_mat[0:2*Nharm, 0:2*Nharm]
    sb_12_mat = sb_mat[0:2*Nharm, 2*Nharm:4*Nharm]
    sb_21_mat = sb_mat[2*Nharm:4*Nharm, 0:2*Nharm]
    sb_22_mat = sb_mat[2*Nharm:4*Nharm, 2*Nharm:4*Nharm]

    d_mat = matmul(sa_12_mat, np.linalg.inv(unit_mat - matmul(sb_11_mat, sa_22_mat)))
    f_mat = matmul(sb_21_mat, np.linalg.inv(unit_mat - matmul(sa_22_mat, sb_11_mat)))

    s_11_mat = sa_11_mat + matmul(d_mat, sb_11_mat, sa_21_mat)
    s_12_mat = matmul(d_mat, sb_12_mat)
    s_21_mat = matmul(f_mat, sa_21_mat)
    s_22_mat = sb_22_mat + matmul(f_mat, sa_22_mat, sb_12_mat)

    s_ret_mat = np.concatenate((np.concatenate((s_11_mat, s_12_mat), axis=1),
                                np.concatenate((s_21_mat, s_22_mat), axis=1)))
    return s_ret_mat


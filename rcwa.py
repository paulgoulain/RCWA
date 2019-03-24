# coding: utf-8

# rcwa method implemented as in
# http://emlab.utep.edu/ee5390cem.htm
# imports and steps 1-3
import numpy as np
#import matplotlib.pyplot as plt
#from mpl_toolkits.axes_grid1 import make_axes_locatable
import cmath
import convmat_func as convmat_func
import tmm_func as tmm_func
import importlib
import argparse
import os
import toml

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('path') 
    args = parser.parse_args()
    if not os.path.exists(args.path):
        raise FileNotFoundError('{} not a valid input file'.format(args.path))

    input_toml = toml.load(args.path)

    # UNITS
    degrees_to_rad = np.pi/180
    # SOURCE PARAMETERS
    norm_lambda = 2*np.pi/input_toml['source']['wavelength']
    K0 = 1
    THETA = input_toml['source']['theta'] * degrees_to_rad
    PHI = input_toml['source']['phi'] * degrees_to_rad
    P_TE = input_toml['source']['te_amplitude'][0] + \
            input_toml['source']['te_amplitude'][1]*1j  # amplitude of TE polarization

    P_TM = input_toml['source']['tm_amplitude'][0] + \
            input_toml['source']['tm_amplitude'][1]*1j  # amplitude of TM polarization
    # permeability in the reflection region
    UR2 = 1.0
    # permittivity in the reflection region
    ER2 = input_toml['superstrate']['epsilon']
    # permeability in the transmission region
    UR1 = 1.0
    # permittivity in the transmission region
    ER1 = input_toml['substrate']['epsilon']
    
    # DEVICE PARAMETERS
    # period in x
    Lx = input_toml['periodicity']['period_x']*norm_lambda
    # period in y
    Ly = input_toml['periodicity']['period_y']*norm_lambda
    # RCWA PARAMETERS
    # number of spatial harmonics along x and y
    P_range = input_toml['periodicity']['harmonics_x']
    Q_range = input_toml['periodicity']['harmonics_y']
    P_high = int(np.floor(P_range/2))
    P_low = -P_high
    Q_high = int(np.floor(Q_range/2))
    Q_low = -Q_high
    P_vec = np.linspace(P_low, P_high, P_range)
    Q_vec = np.linspace(Q_low, Q_high, Q_range)

    # BUILD DEVICE LAYERS ON HIGH RESOLUTION GRID
    #number of point along x in real-space grid
    Nx = 512
    #number of point along y in real-space grid
    Ny = int(np.ceil((Nx*Ly/Lx)))
    
    num_layers = len(input_toml['layer'])
    L = [None]*num_layers
    er_vec = [None]*num_layers
    ur_vec = [None]*num_layers
    erc_vec = [None]*num_layers
    urc_vec = [None]*num_layers
    for i in range(0, num_layers):
        L[i] = input_toml['layer'][i]['thickness']*norm_lambda
        ur_vec[i] = 1.0*np.ones((Nx, Ny))
        epsilon = input_toml['layer'][i]['epsilon']
        if type(epsilon) == float or type(epsilon) == int:
            er_vec[i] = epsilon*np.ones((Nx, Ny))
        elif os.path.exists(epsilon):
            er_vec[i] = np.loadtxt(epsilon, delimiter=',')
        else:
            raise ValueError('Invalid epsilon for layer {} - should be a float or a path to .csv file'.format(i))

    for i in range(0, num_layers):
        erc_vec[i] = convmat_func.convmat(er_vec[i], P_range, Q_range)
        urc_vec[i] = convmat_func.convmat(ur_vec[i], P_range, Q_range)

    # TODO fix vectorisation in convmat
    '''
    fig, (ax1, ax2) = plt.subplots(1, 2)

    im1 = ax1.matshow(UR)
    div1 = make_axes_locatable(ax1)
    cax1 = div1.append_axes('right', size='20%', pad=0.05)
    plt.colorbar(im1, cax=cax1)

    im2 = ax2.matshow(ER)
    div2 = make_axes_locatable(ax2)
    cax2 = div2.append_axes('right', size='20%', pad=0.05)
    plt.colorbar(im2, cax=cax2)

    fig, (ax3, ax4) = plt.subplots(1, 2)

    im3 = ax3.matshow(np.real(URC1))
    div3 = make_axes_locatable(ax3)
    cax3 = div3.append_axes('right', size='20%', pad=0.05)
    plt.colorbar(im3, cax=cax3)

    im4 = ax4.matshow(np.real(ERC1))
    div4 = make_axes_locatable(ax4)
    cax4 = div4.append_axes('right', size='20%', pad=0.05)
    plt.colorbar(im4, cax=cax4)
    '''

    # initialise parameters for the computation
    nr1 = np.sqrt(UR1*ER1)
    k_inc = K0*nr1*np.array(([np.sin(THETA)*np.cos(PHI), np.sin(THETA)*np.sin(PHI), np.cos(THETA)]))
    k_inc /= K0
    k_ref = np.array(([k_inc[0], k_inc[1], -k_inc[2]]))
    nr2 = np.sqrt(UR2*ER2)
    k_trn = np.array(([k_ref[0], k_ref[1], 0]))
    k_trn[2] = np.sqrt(K0*K0*nr2*nr2 - k_trn[0]*k_trn[0] - k_trn[1]*k_trn[1])

    Nharm = P_range*Q_range
    kx_mat = np.zeros((Nharm, Nharm), dtype=complex)
    ky_mat = np.zeros((Nharm, Nharm), dtype=complex)
    kz_0_mat = np.zeros((Nharm, Nharm), dtype=complex)
    kz_ref_mat = np.zeros((Nharm, Nharm), dtype=complex)
    kz_trn_mat = np.zeros((Nharm, Nharm), dtype=complex)
    # TODO vectorise + nicify
    count_i = -1
    for i in P_vec:
        count_i += 1
        count_j = -1
        for j in Q_vec:
            count_j += 1
            pos = count_i*P_range + count_j
            kx_mat[pos, pos] = k_inc[0] - j*2*np.pi/(Lx*K0)
            ky_mat[pos, pos] = k_inc[1] - i*2*np.pi/(Ly*K0)
            kz_0_mat[pos, pos] = np.conj(np.lib.scimath.sqrt(1-(kx_mat[pos,pos])**2-(ky_mat[pos,pos])**2))
            kz_ref_mat[pos, pos] = -np.conj(np.lib.scimath.sqrt(np.conj(ER1)*np.conj(UR1)-kx_mat[pos,pos]**2-ky_mat[pos,pos]**2))
            kz_trn_mat[pos, pos] = np.conj(np.lib.scimath.sqrt(np.conj(ER2)*np.conj(UR2)-kx_mat[pos,pos]**2-ky_mat[pos,pos]**2))

    I_mat = np.diag(np.ones(Nharm))
    I_big_mat = np.diag(np.ones(2*Nharm))
    zeros_mat = np.zeros((Nharm, Nharm))
    zeros_big_mat = np.zeros((2*Nharm, 2*Nharm))
    W0_mat = np.diag(np.ones(2*Nharm))
    P0_mat = np.zeros((2*Nharm, 2*Nharm), dtype=complex)
    P0_mat[0:Nharm, 0:Nharm] = tmm_func.matmul(kx_mat, ky_mat)
    P0_mat[0:Nharm, Nharm:2*Nharm] = I_mat - tmm_func.matmul(kx_mat,kx_mat)
    P0_mat[Nharm:2*Nharm, 0:Nharm] = tmm_func.matmul(ky_mat,ky_mat)-I_mat
    P0_mat[Nharm:2*Nharm, Nharm:2*Nharm] = -tmm_func.matmul(ky_mat,kx_mat)
    Q0_mat = P0_mat
    lambda0_mat = np.concatenate((np.concatenate((1j*kz_0_mat, zeros_mat), axis = 1),
                                   np.concatenate((zeros_mat, 1j*kz_0_mat), axis = 1)))
    V0_mat = tmm_func.matmul(Q0_mat, np.linalg.inv(lambda0_mat))
        
    S_global = np.concatenate((np.concatenate((zeros_big_mat, I_big_mat), axis = 1),
                                   np.concatenate((I_big_mat, zeros_big_mat), axis = 1)))

    # iterating through layers
    for i in range(0, num_layers):
        ERC = erc_vec[i]
        URC = urc_vec[i]
        thickness = L[i]

        P_mat = np.zeros((2*Nharm, 2*Nharm), dtype = complex)
        P_mat[0:Nharm, 0:Nharm] = tmm_func.matmul(kx_mat, np.linalg.inv(ERC), ky_mat)
        P_mat[0:Nharm, Nharm:2*Nharm] = URC - tmm_func.matmul(kx_mat, np.linalg.inv(ERC), kx_mat)
        P_mat[Nharm:2*Nharm, 0:Nharm] = tmm_func.matmul(ky_mat, np.linalg.inv(ERC), ky_mat) - URC
        P_mat[Nharm:2*Nharm, Nharm:2*Nharm] = -tmm_func.matmul(ky_mat, np.linalg.inv(ERC), kx_mat)

        Q_mat = np.zeros((2*Nharm, 2*Nharm), dtype = complex)
        Q_mat[0:Nharm, 0:Nharm] = tmm_func.matmul(kx_mat, np.linalg.inv(URC), ky_mat)
        Q_mat[0:Nharm, Nharm:2*Nharm] = ERC - tmm_func.matmul(kx_mat, np.linalg.inv(URC), kx_mat)
        Q_mat[Nharm:2*Nharm, 0:Nharm] = tmm_func.matmul(ky_mat, np.linalg.inv(URC), ky_mat) - ERC
        Q_mat[Nharm:2*Nharm, Nharm:2*Nharm] = -tmm_func.matmul(ky_mat, np.linalg.inv(URC), kx_mat)

        Omega_sq_mat = tmm_func.matmul(P_mat, Q_mat)
        v_vec, W_mat = np.linalg.eig(Omega_sq_mat)
        lambda_mat = np.diag(np.conj(np.sqrt(v_vec)))
        V_mat = tmm_func.matmul(Q_mat, W_mat, np.linalg.inv(lambda_mat))
        A_mat = tmm_func.matmul(np.linalg.inv(W_mat), W0_mat) + tmm_func.matmul(np.linalg.inv(V_mat), V0_mat)
        B_mat = tmm_func.matmul(np.linalg.inv(W_mat), W0_mat) - tmm_func.matmul(np.linalg.inv(V_mat), V0_mat)
        X_mat = np.diag(np.exp(-np.diag(lambda_mat)*K0*thickness))
        S_11 = tmm_func.matmul(np.linalg.inv(A_mat - tmm_func.matmul(X_mat, B_mat, np.linalg.inv(A_mat), X_mat, B_mat)),
            (tmm_func.matmul(X_mat, B_mat, np.linalg.inv(A_mat), X_mat, A_mat)\
            - B_mat))
        S_12 = tmm_func.matmul(np.linalg.inv(A_mat - tmm_func.matmul(X_mat, B_mat, np.linalg.inv(A_mat), X_mat, B_mat)),
            X_mat, (A_mat - tmm_func.matmul(B_mat, np.linalg.inv(A_mat), B_mat)))
        S = np.concatenate((np.concatenate((S_11, S_12), axis = 1),
                            np.concatenate((S_12, S_11), axis = 1)))
        
        S_global = tmm_func.redheffer_star_prod(S_global, S, I_big_mat)

    # reflection region
    Q_ref_mat = np.zeros((2*Nharm, 2*Nharm), dtype = complex)
    Q_ref_mat[0:Nharm, 0:Nharm] = tmm_func.matmul(kx_mat, ky_mat)
    Q_ref_mat[0:Nharm, Nharm:2*Nharm] = UR1*ER1*I_mat - tmm_func.matmul(kx_mat, kx_mat)
    Q_ref_mat[Nharm:2*Nharm, 0:Nharm] = tmm_func.matmul(ky_mat, ky_mat) - UR1*ER1*I_mat
    Q_ref_mat[Nharm:2*Nharm, Nharm:2*Nharm] = -tmm_func.matmul(ky_mat, kx_mat)
    Q_ref_mat /= UR1

    W_ref_mat = np.diag(np.ones(2*Nharm))

    lambda_ref_mat = np.concatenate((np.concatenate((-1j*kz_ref_mat, zeros_mat), axis = 1),
                            np.concatenate((zeros_mat, -1j*kz_ref_mat), axis = 1)))
    V_ref_mat = tmm_func.matmul(Q_ref_mat, np.linalg.inv(lambda_ref_mat))
    A_ref_mat = tmm_func.matmul(np.linalg.inv(W0_mat), W_ref_mat) + tmm_func.matmul(np.linalg.inv(V0_mat), V_ref_mat)
    B_ref_mat = tmm_func.matmul(np.linalg.inv(W0_mat), W_ref_mat) - tmm_func.matmul(np.linalg.inv(V0_mat), V_ref_mat)
    S_ref_11 = -tmm_func.matmul(np.linalg.inv(A_ref_mat), B_ref_mat)
    S_ref_12 = 2*np.linalg.inv(A_ref_mat)
    S_ref_21 = 0.5*(A_ref_mat - tmm_func.matmul(B_ref_mat, np.linalg.inv(A_ref_mat), B_ref_mat))
    S_ref_22 = tmm_func.matmul(B_ref_mat, np.linalg.inv(A_ref_mat))
    S_ref = np.concatenate((np.concatenate((S_ref_11, S_ref_12), axis = 1),
                            np.concatenate((S_ref_21, S_ref_22), axis = 1)))

    S_global = tmm_func.redheffer_star_prod(S_ref, S_global, I_big_mat)

    # transmission region
    Q_trn_mat = np.zeros((2*Nharm, 2*Nharm), dtype = complex)
    Q_trn_mat[0:Nharm, 0:Nharm] = tmm_func.matmul(kx_mat, ky_mat)
    Q_trn_mat[0:Nharm, Nharm:2*Nharm] = UR2*ER2*I_mat - tmm_func.matmul(kx_mat, kx_mat)
    Q_trn_mat[Nharm:2*Nharm, 0:Nharm] = tmm_func.matmul(ky_mat, ky_mat) - UR2*ER2*I_mat
    Q_trn_mat[Nharm:2*Nharm, Nharm:2*Nharm] = -tmm_func.matmul(ky_mat, kx_mat)
    Q_trn_mat /= UR2

    W_trn_mat = np.diag(np.ones(2*Nharm))
    lambda_trn_mat = np.concatenate((np.concatenate((1j*kz_trn_mat, zeros_mat), axis = 1),
                            np.concatenate((zeros_mat, 1j*kz_trn_mat), axis = 1)))
    V_trn_mat = tmm_func.matmul(Q_trn_mat, np.linalg.inv(lambda_trn_mat))
    A_trn_mat = tmm_func.matmul(np.linalg.inv(W0_mat), W_trn_mat) + tmm_func.matmul(np.linalg.inv(V0_mat), V_trn_mat)
    B_trn_mat = tmm_func.matmul(np.linalg.inv(W0_mat), W_trn_mat) - tmm_func.matmul(np.linalg.inv(V0_mat), V_trn_mat)

    S_trn_11 = tmm_func.matmul(B_trn_mat, np.linalg.inv(A_trn_mat))
    S_trn_12 = 0.5*(A_trn_mat - tmm_func.matmul(B_trn_mat, np.linalg.inv(A_trn_mat), B_trn_mat))
    S_trn_21 = 2*np.linalg.inv(A_trn_mat)
    S_trn_22 = -tmm_func.matmul(np.linalg.inv(A_trn_mat), B_trn_mat)
    S_trn = np.concatenate((np.concatenate((S_trn_11, S_trn_12), axis = 1),
                            np.concatenate((S_trn_21, S_trn_22), axis = 1)))
    S_global = tmm_func.redheffer_star_prod(S_global, S_trn, I_big_mat)

    # computing input and output fields
    delta_vec = np.zeros(Nharm)
    delta_vec[int(np.floor(Nharm/2))] = 1
    z_unit_vec = np.array(([0, 0, 1]))
    alpha_TE = np.cross(z_unit_vec, k_inc)
    alpha_TE = alpha_TE/np.linalg.norm(alpha_TE)
    alpha_TM = np.cross(alpha_TE, k_inc)
    alpha_TM = alpha_TM/np.linalg.norm(alpha_TM)
    E_inc = np.zeros(2*Nharm, dtype=complex)
    E_inc[0:Nharm] = (P_TM*alpha_TM + P_TE*alpha_TE)[0]*delta_vec
    E_inc[Nharm:2*Nharm] = (P_TM*alpha_TM + P_TE*alpha_TE)[1]*delta_vec
    c_inc = tmm_func.matmul(np.linalg.inv(W_ref_mat), E_inc)
    E_ref_xy = tmm_func.matmul(W_ref_mat, S_global[0:2*Nharm, 0:2*Nharm], c_inc)
    E_trn_xy = tmm_func.matmul(W_trn_mat, S_global[2*Nharm:4*Nharm, 0:2*Nharm], c_inc)
    E_ref_z = -tmm_func.matmul(np.linalg.inv(kz_ref_mat), tmm_func.matmul(kx_mat, E_ref_xy[0:Nharm]) + tmm_func.matmul(ky_mat, E_ref_xy[Nharm:2*Nharm]))
    E_trn_z = -tmm_func.matmul(np.linalg.inv(kz_trn_mat), tmm_func.matmul(kx_mat, E_trn_xy[0:Nharm]) + tmm_func.matmul(ky_mat, E_trn_xy[Nharm:2*Nharm]))

    r_sq = np.zeros((Nharm), dtype=complex)
    t_sq = np.zeros((Nharm), dtype=complex)
    for i in range(0, Nharm):
        r_sq[i] = E_ref_xy[i]*np.conj(E_ref_xy[i]) + E_ref_xy[Nharm+i]*np.conj(E_ref_xy[Nharm+i]) + E_ref_z[i]*np.conj(E_ref_z[i])
        t_sq[i] = E_trn_xy[i]*np.conj(E_trn_xy[i]) + E_trn_xy[Nharm+i]*np.conj(E_trn_xy[Nharm+i]) + E_trn_z[i]*np.conj(E_trn_z[i])

    R = -np.real(tmm_func.matmul(kz_ref_mat, r_sq)/UR1)
    R /= np.real(k_inc[2]/UR1)#

    T = np.real(tmm_func.matmul(kz_trn_mat, t_sq)/UR2)
    T /= np.real(k_inc[2]/UR2)

    np.set_printoptions(precision=4)

    with open ('output.toml', 'w') as fid:
        fid.write('[R]\n')
        for i in range(0, Nharm):
            fid.write('{:.0f}{:.0f} = '.format(P_vec[i%P_range], Q_vec[i//P_range]) + '{:.4f}\n'.format(R[i]))
        fid.write('sum = {:.4f}\n'.format(np.sum(R)))
        fid.write('\n[T]\n')
        for i in range(0, Nharm):
            fid.write('{:.0f}{:.0f} = '.format(P_vec[i%P_range], Q_vec[i//P_range]) + '{:.4f}\n'.format(T[i]))
        fid.write('sum = {:.4f}\n'.format(np.sum(T)))
        fid.write('\n[R_T]\n')
        fid.write('sum = {:.4f}'.format(np.sum(R+T)))

if __name__ == '__main__':
    main()

# coding: utf-8

# transfer matrix method implemented as python
# based on the notes byhttp://emlab.utep.edu/ee5390cem.htm
import numpy as np
import argparse
import os
import toml
import cmath

def calc_gap_layer_params(kx, ky):
    ur = 1
    er = 1
    Q = np.array(([kx*ky, ur*er+ky*ky],[-(ur*er+kx*kx), -kx*ky]))/ur
    V = -1j*Q
    return V

def calc_layer_params(ur, er, kx, ky):
    Q = np.array(([kx*ky, ur*er-kx*kx],[ky*ky-ur*er, -kx*ky]))/ur
    kz = cmath.sqrt(ur*er-kx*kx-ky*ky)
    Omega = 1j*kz*np.array(([1, 0], [0, 1]))
    V = np.matmul(Q, np.linalg.inv(Omega))
    return Omega, V
    
def init_global_S_mat():
    S_global = np.array(([0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]))
    return S_global

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

def calc_S_mat(L, ur, er, kx, ky, I, K0):
    Omegai, Vi = calc_layer_params(ur, er, kx, ky)
    Vg = calc_gap_layer_params(kx, ky)
    Ai = I + np.matmul(np.linalg.inv(Vi), Vg)
    Bi = I - np.matmul(np.linalg.inv(Vi), Vg)
    Xi = np.diag(np.exp(np.diag(Omegai)*K0*L))
    Ai_inv = np.linalg.inv(Ai)
    Di = Ai - matmul(Xi, Bi, Ai_inv, Xi, Bi)
    Di_inv = np.linalg.inv(Di)
    S_11 = matmul(Di_inv, matmul(Xi, Bi, Ai_inv, Xi, Ai) - Bi)
    S_12 = matmul(Di_inv, Xi, Ai - matmul(Bi, Ai_inv, Bi))
    # S_12 = S_21, S_11 = S_22
    S = np.concatenate((np.concatenate((S_11, S_12), axis = 1),
                               np.concatenate((S_12, S_11), axis = 1)))
    return S

def redheffer_star_prod(Sa_4x4, Sb_4x4, I):
    Sa_11 = Sa_4x4[np.ix_([0, 1], [0, 1])]
    Sa_12 = Sa_4x4[np.ix_([0, 1], [2, 3])]
    Sa_21 = Sa_4x4[np.ix_([2, 3], [0, 1])]
    Sa_22 = Sa_4x4[np.ix_([2, 3], [2, 3])]
    
    Sb_11 = Sb_4x4[np.ix_([0, 1], [0, 1])]
    Sb_12 = Sb_4x4[np.ix_([0, 1], [2, 3])]
    Sb_21 = Sb_4x4[np.ix_([2, 3], [0, 1])]
    Sb_22 = Sb_4x4[np.ix_([2, 3], [2, 3])]
    
    D = matmul(Sa_12, np.linalg.inv(I - matmul(Sb_11, Sa_22)))
    F = matmul(Sb_21, np.linalg.inv(I - matmul(Sa_22, Sb_11)))
    
    S_11 = Sa_11 + matmul(D, Sb_11, Sa_21)
    S_12 = matmul(D, Sb_12)
    S_21 = matmul(F, Sa_21)
    S_22 = Sb_22 + matmul(F, Sa_22, Sb_12)
    
    S_ret = np.concatenate((np.concatenate((S_11, S_12), axis = 1),
                               np.concatenate((S_21, S_22), axis = 1)))
    return S_ret

def calc_S_ref(ur, er, kx, ky, I):
    Omega_ref, V_ref = calc_layer_params(ur, er, kx, ky)
    Vg = calc_gap_layer_params(kx, ky)
    A_ref = I + matmul(np.linalg.inv(Vg), V_ref)
    B_ref = I - matmul(np.linalg.inv(Vg), V_ref)
    S_ref_11 = -matmul(np.linalg.inv(A_ref), B_ref)
    S_ref_12 = 2*np.linalg.inv(A_ref)
    S_ref_21 = 0.5*(A_ref - matmul(B_ref, np.linalg.inv(A_ref), B_ref))
    S_ref_22 = matmul(B_ref, np.linalg.inv(A_ref))
    S_ref = np.concatenate((np.concatenate((S_ref_11, S_ref_12), axis = 1),
                                np.concatenate((S_ref_21, S_ref_22), axis = 1)))
    return S_ref

def calc_S_trn(ur, er, kx, ky, I):
    Omega_trn, V_trn = calc_layer_params(ur, er, kx, ky)
    Vg = calc_gap_layer_params(kx, ky)
    A_trn = I + matmul(np.linalg.inv(Vg), V_trn)
    B_trn = I - matmul(np.linalg.inv(Vg), V_trn)
    S_trn_11 =  matmul(B_trn, np.linalg.inv(A_trn))
    S_trn_12 = 0.5*(A_trn - matmul(B_trn, np.linalg.inv(A_trn), B_trn))
    S_trn_21 = 2*np.linalg.inv(A_trn)
    S_trn_22 = -matmul(np.linalg.inv(A_trn), B_trn)
    S_trn = np.concatenate((np.concatenate((S_trn_11, S_trn_12), axis = 1),
                                   np.concatenate((S_trn_21, S_trn_22), axis = 1)))
    return S_trn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    args = parser.parse_args()
    if not os.path.exists(args.path):
        raise FileNotFoundError('{} not a valid input file'.format(args.path))

    input_toml = toml.load(args.path)
    #print(input_toml)
    # UNITS
    DEGREES = np.pi/180

    # SOURCE PARAMETERS # TODO test for deviations from lambda = 2*pi
    norm_lambda = 2*np.pi/input_toml['source']['wavelength']
    # free space wavevector; layer thickness normalised below
    K0 = 1
    I = np.array(([1, 0], [0, 1])) # 2x2 identity matrix used in many places
    THETA = input_toml['source']['theta'] * DEGREES #elevation angle
    PHI = input_toml['source']['phi'] * DEGREES #azimuthal angle
    P_TE = input_toml['source']['te_amplitude'][0] + \
            input_toml['source']['te_amplitude'][1]*1j #amplitude of TE polarization
    P_TM = input_toml['source']['tm_amplitude'][0] + \
            input_toml['source']['tm_amplitude'][1]*1j #amplitude of TM polarization
    # normalise polarisation
    norm_pol = np.sqrt((np.real(P_TE))**2 + (np.imag(P_TE))**2 + \
    (np.real(P_TM))**2 + (np.imag(P_TM))**2)
    P_TM = P_TM/norm_pol
    P_TE = P_TE/norm_pol

    # EXTERNAL MATERIALS
    #permeability in the reflection region
    UR1 = input_toml['superstrate']['mu']
    #permittivity in the reflection region
    ER1 = input_toml['superstrate']['epsilon']
    #permeability in the transmission region
    UR2 = input_toml['substrate']['mu']
    #permittivity in the transmission region
    ER2 = input_toml['substrate']['epsilon']

    # DEFINE LAYERS
    num_layers = len(input_toml['layer'])
    UR = [None]*num_layers
    ER = [None]*num_layers
    L = [None]*num_layers
    for i in range(0, num_layers):
        UR[i] = input_toml['layer'][i]['mu']
        ER[i] = input_toml['layer'][i]['epsilon']
        L[i] = input_toml['layer'][i]['thickness']*norm_lambda

    nr1 = np.sqrt(UR1*ER1)
    k_inc = K0*nr1*np.array(([np.sin(THETA)*np.cos(PHI), np.sin(THETA)*np.sin(PHI), np.cos(THETA)]))
    kx, ky = k_inc[0], k_inc[1]
    z_unit_vec = np.array(([0, 0, 1]))
    alpha_TE = np.cross(z_unit_vec, k_inc)
    alpha_TE = alpha_TE/np.linalg.norm(alpha_TE)
    alpha_TM = np.cross(alpha_TE, k_inc)
    alpha_TM = alpha_TM/np.linalg.norm(alpha_TM)
    E_inc = P_TM*alpha_TM + P_TE*alpha_TE
    c_inc = np.array(([E_inc[0], E_inc[1], 0, 0]))

    S_global = init_global_S_mat()
    # calc device S
    for i in range(0, len(UR)):
        ur = UR[i]
        er = ER[i]
        l = L[i]
        S_layer = calc_S_mat(l, ur, er, kx, ky, I, K0)
        S_global = redheffer_star_prod(S_global, S_layer, I)

    # calc refl S
    S_global = redheffer_star_prod(calc_S_ref(UR1, ER1, kx, ky, I), S_global, I)

    # calc trn S
    S_global = redheffer_star_prod(S_global, calc_S_trn(UR2, ER2, kx, ky, I), I)

    c_ret = matmul(S_global, c_inc)
    E_ref = np.array(([c_ret[0], c_ret[1], 0]))
    E_ref[2] = -(k_inc[0]*E_ref[0]+k_inc[1]*E_ref[1])/k_inc[2]
    k_trn = np.array(([k_inc[0], k_inc[1], 0]))
    nr2 = np.sqrt(ER2*UR2)
    k_trn[2] = np.sqrt(K0*K0*nr2*nr2 - k_trn[0]*k_trn[0] - k_trn[1]*k_trn[1])
    E_trn = np.array(([c_ret[2], c_ret[3], 0]))
    E_trn[2] = -(k_trn[0]*E_trn[0]+k_trn[1]*E_trn[1])/(k_trn[2])
    R = round((np.linalg.norm(E_ref))**2, 4)
    T = round((np.linalg.norm(E_trn))**2*((k_trn[2]/UR2).real)/((k_inc[2]/UR1).real), 4)

    print({'R': R, 'T': T, 'R+T': R+T})
    with open('output.toml', 'w') as fid:
        output = {'output': {r'R': R, r'T': T, r'R+T': R+T}}
        toml.dump(output, fid)

if __name__ == '__main__':
    main()



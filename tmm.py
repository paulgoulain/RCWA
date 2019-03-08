# coding: utf-8

# transfer matrix method implemented as python
# based on the notes byhttp://emlab.utep.edu/ee5390cem.htm
import argparse
import os
import cmath

import toml
import numpy as np
#TODO add docstrings
def calc_gap_layer_params(kx, ky):
    ur = 1
    er = 1
    q_mat = np.array(([kx*ky, ur*er+ky*ky], [-(ur*er+kx*kx), -kx*ky]))/ur
    v_mat = -1j*q_mat
    return v_mat

def calc_layer_params(ur, er, kx, ky):
    q_mat = np.array(([kx*ky, ur*er-kx*kx], [ky*ky-ur*er, -kx*ky]))/ur
    kz = cmath.sqrt(ur*er-kx*kx-ky*ky)
    omega_mat = 1j*kz*np.array(([1, 0], [0, 1]))
    v_mat = np.matmul(q_mat, np.linalg.inv(omega_mat))
    return omega_mat, v_mat

def init_global_s_mat():
    s_global = np.array(([0, 0, 1, 0], [0, 0, 0, 1],
                         [1, 0, 0, 0], [0, 1, 0, 0]))
    return s_global

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

def calc_s_mat(layer_thickness, ur, er, kx, ky, unit_mat, k0):
    omegai_mat, vi_mat = calc_layer_params(ur, er, kx, ky)
    vg_mat = calc_gap_layer_params(kx, ky)
    ai_mat = unit_mat + np.matmul(np.linalg.inv(vi_mat), vg_mat)
    bi_mat = unit_mat - np.matmul(np.linalg.inv(vi_mat), vg_mat)
    xi_mat = np.diag(np.exp(np.diag(omegai_mat)*k0*layer_thickness))
    ai_inv_mat = np.linalg.inv(ai_mat)
    di_mat = ai_mat - matmul(xi_mat, bi_mat, ai_inv_mat, xi_mat, bi_mat)
    di_inv_mat = np.linalg.inv(di_mat)
    s_11_mat = matmul(di_inv_mat, matmul(
        xi_mat, bi_mat, ai_inv_mat, xi_mat, ai_mat) - bi_mat)
    s_12_mat = matmul(di_inv_mat, xi_mat, ai_mat
                      - matmul(bi_mat, ai_inv_mat, bi_mat))
    # S_12 = S_21, S_11 = S_22
    s_mat = np.concatenate((np.concatenate((s_11_mat, s_12_mat), axis=1),
                            np.concatenate((s_12_mat, s_11_mat), axis=1)))
    return s_mat

def redheffer_star_prod(sa_4x4_mat, sb_4x4_mat, unit_mat):
    sa_11_mat = sa_4x4_mat[np.ix_([0, 1], [0, 1])]
    sa_12_mat = sa_4x4_mat[np.ix_([0, 1], [2, 3])]
    sa_21_mat = sa_4x4_mat[np.ix_([2, 3], [0, 1])]
    sa_22_mat = sa_4x4_mat[np.ix_([2, 3], [2, 3])]

    sb_11_mat = sb_4x4_mat[np.ix_([0, 1], [0, 1])]
    sb_12_mat = sb_4x4_mat[np.ix_([0, 1], [2, 3])]
    sb_21_mat = sb_4x4_mat[np.ix_([2, 3], [0, 1])]
    sb_22_mat = sb_4x4_mat[np.ix_([2, 3], [2, 3])]

    d_mat = matmul(sa_12_mat, np.linalg.inv(unit_mat - matmul(sb_11_mat, sa_22_mat)))
    f_mat = matmul(sb_21_mat, np.linalg.inv(unit_mat - matmul(sa_22_mat, sb_11_mat)))

    s_11_mat = sa_11_mat + matmul(d_mat, sb_11_mat, sa_21_mat)
    s_12_mat = matmul(d_mat, sb_12_mat)
    s_21_mat = matmul(f_mat, sa_21_mat)
    s_22_mat = sb_22_mat + matmul(f_mat, sa_22_mat, sb_12_mat)

    s_ret_mat = np.concatenate((np.concatenate((s_11_mat, s_12_mat), axis=1),
                                np.concatenate((s_21_mat, s_22_mat), axis=1)))
    return s_ret_mat

def calc_s_ref(ur, er, kx, ky, unit_mat):
    omega_ref_mat, v_ref_mat = calc_layer_params(ur, er, kx, ky)
    vg_mat = calc_gap_layer_params(kx, ky)
    a_ref_mat = unit_mat + matmul(np.linalg.inv(vg_mat), v_ref_mat)
    b_ref_mat = unit_mat - matmul(np.linalg.inv(vg_mat), v_ref_mat)
    s_ref_11_mat = -matmul(np.linalg.inv(a_ref_mat), b_ref_mat)
    s_ref_12_mat = 2*np.linalg.inv(a_ref_mat)
    s_ref_21_mat = 0.5*(a_ref_mat
                        - matmul(b_ref_mat, np.linalg.inv(a_ref_mat), b_ref_mat))
    s_ref_22_mat = matmul(b_ref_mat, np.linalg.inv(a_ref_mat))
    s_ref_mat = np.concatenate((
        np.concatenate((s_ref_11_mat, s_ref_12_mat), axis=1),
        np.concatenate((s_ref_21_mat, s_ref_22_mat), axis=1)))
    return s_ref_mat

def calc_s_trn(ur, er, kx, ky, unit_mat):
    omega_trn_mat, v_trn_mat = calc_layer_params(ur, er, kx, ky)
    vg_mat = calc_gap_layer_params(kx, ky)
    a_trn_mat = unit_mat + matmul(np.linalg.inv(vg_mat), v_trn_mat)
    b_trn_mat = unit_mat - matmul(np.linalg.inv(vg_mat), v_trn_mat)
    s_trn_11_mat = matmul(b_trn_mat, np.linalg.inv(a_trn_mat))
    s_trn_12_mat = 0.5*(a_trn_mat - matmul(b_trn_mat,
                                           np.linalg.inv(a_trn_mat), b_trn_mat))
    s_trn_21_mat = 2*np.linalg.inv(a_trn_mat)
    s_trn_22_mat = -matmul(np.linalg.inv(a_trn_mat), b_trn_mat)
    s_trn_mat = np.concatenate((
        np.concatenate((s_trn_11_mat, s_trn_12_mat), axis=1),
        np.concatenate((s_trn_21_mat, s_trn_22_mat), axis=1)))
    return s_trn_mat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    args = parser.parse_args()
    if not os.path.exists(args.path):
        raise FileNotFoundError('{} not a valid input file'.format(args.path))

    input_toml = toml.load(args.path)
    #print(input_toml)
    degrees_to_rad = np.pi/180

    # source parameters
    norm_lambda = 2*np.pi/input_toml['source']['wavelength']
    # free space wavevector; layer thickness normalised below
    k0 = 1
    unit_mat = np.array(([1, 0], [0, 1]))  # 2x2 identity matrix used in many places
    theta = input_toml['source']['theta'] * degrees_to_rad  # elevation angle
    phi = input_toml['source']['phi'] * degrees_to_rad  # azimuthal angle
    p_te = input_toml['source']['te_amplitude'][0] + \
            input_toml['source']['te_amplitude'][1]*1j  # amplitude of TE polarization
    p_tm = input_toml['source']['tm_amplitude'][0] + \
            input_toml['source']['tm_amplitude'][1]*1j  # amplitude of TM polarization
    # normalise polarisation
    norm_pol = np.sqrt((np.real(p_te))**2 + (np.imag(p_te))**2 + \
    (np.real(p_tm))**2 + (np.imag(p_tm))**2)
    p_tm = p_tm/norm_pol
    p_te = p_te/norm_pol

    # permeability in the reflection region
    ur1 = input_toml['superstrate']['mu']
    # permittivity in the reflection region
    er1 = input_toml['superstrate']['epsilon']
    # permeability in the transmission region
    ur2 = input_toml['substrate']['mu']
    # permittivity in the transmission region
    er2 = input_toml['substrate']['epsilon']

    # define layers
    num_layers = len(input_toml['layer'])
    ur_vec = [None]*num_layers
    er_vec = [None]*num_layers
    layer_thicknesses_vec = [None]*num_layers
    for i in range(0, num_layers):
        ur_vec[i] = input_toml['layer'][i]['mu']
        er_vec[i] = input_toml['layer'][i]['epsilon']
        layer_thicknesses_vec[i] = input_toml['layer'][i]['thickness']*norm_lambda

    nr1 = np.sqrt(ur1*er1)
    k_inc = k0*nr1*np.array(([np.sin(theta)*np.cos(phi),
                              np.sin(theta)*np.sin(phi), np.cos(theta)]))
    kx, ky = k_inc[0], k_inc[1]
    z_unit_vec = np.array(([0, 0, 1]))
    alpha_te = np.cross(z_unit_vec, k_inc)
    alpha_te = alpha_te/np.linalg.norm(alpha_te)
    alpha_tm = np.cross(alpha_te, k_inc)
    alpha_tm = alpha_tm/np.linalg.norm(alpha_tm)
    E_inc = p_tm*alpha_tm + p_te*alpha_te
    c_inc = np.array(([E_inc[0], E_inc[1], 0, 0]))

    s_global_mat = init_global_s_mat()
    # take layers into account
    for i in range(0, num_layers):
        ur = ur_vec[i]
        er = er_vec[i]
        l = layer_thicknesses_vec[i]
        s_layer_mat = calc_s_mat(l, ur, er, kx, ky, unit_mat, k0)
        s_global_mat = redheffer_star_prod(s_global_mat, s_layer_mat, unit_mat)

    # take superstrate into account
    s_global_mat = redheffer_star_prod(calc_s_ref(ur1, er1, kx, ky, unit_mat),
                                       s_global_mat, unit_mat)

    # take substrate into account
    s_global_mat = redheffer_star_prod(s_global_mat,
                                       calc_s_trn(ur2, er2, kx, ky, unit_mat), unit_mat)

    c_ret = matmul(s_global_mat, c_inc)
    e_field_ref = np.array(([c_ret[0], c_ret[1], 0]))
    e_field_ref[2] = -(k_inc[0]*e_field_ref[0]+k_inc[1]*e_field_ref[1])/k_inc[2]
    k_trn = np.array(([k_inc[0], k_inc[1], 0]))
    nr2 = np.sqrt(er2*ur2)
    k_trn[2] = np.sqrt(k0*k0*nr2*nr2 - k_trn[0]*k_trn[0] - k_trn[1]*k_trn[1])
    e_field_trn = np.array(([c_ret[2], c_ret[3], 0]))
    e_field_trn[2] = -(k_trn[0]*e_field_trn[0]+k_trn[1]*e_field_trn[1])\
        /(k_trn[2])
    ref_eff = round((np.linalg.norm(e_field_ref))**2, 4)
    trn_eff = round((np.linalg.norm(e_field_trn))**2*((k_trn[2]/ur2).real)\
            /((k_inc[2]/ur1).real), 4)

    print({'ref_eff': ref_eff, 'trn_eff': trn_eff,
           'ref_eff+trn_eff': ref_eff+trn_eff})
    with open('output.toml', 'w') as fid:
        output = {'output': {r'ref_eff': ref_eff, 'trn_eff': trn_eff,
                             'ref_eff+trn_eff': ref_eff+trn_eff}}
        toml.dump(output, fid)

if __name__ == '__main__':
    main()
